#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import hashlib
import io
import os
import re
import socket
from enum import Enum
from typing import Any, Dict, List, Union

import boto3
import botocore
import torch

from internlm.utils.common import SingletonMeta
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

boto3_url_re = re.compile(r"([^\.]+)\.([\d\.]+)")

MB = 1024**2

storage_manager = None


def check_folder(fp: str):
    storage_manager.assert_fp_exists(fp)


def get_fns(fp: str):
    return storage_manager.get_fns(fp)


def llm_load(fp: str, *args, **kwargs):
    return storage_manager.load(fp, *args, **kwargs)


def llm_save(save_path: str, saved_obj: Any, *args, **kwargs):
    storage_manager.save(save_path, *args, saved_obj=saved_obj, **kwargs)


class CheckpointType(Enum):
    NORMAL_CHECKPOINT = 1


class StorageClient:
    """
    StorageClient as a client for s3 storage access.
    """

    def __init__(self, handler) -> None:
        self.handler = handler

    @staticmethod
    def load(client, load_path: str, map_location):
        raise NotImplementedError

    @staticmethod
    def sync_upload_fileobj(*args, saved_obj=None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def assert_fp_exists(client):
        raise NotImplementedError

    @staticmethod
    def get_fns(client):
        raise NotImplementedError


class Boto3MetaInfo:
    def __init__(self, client: StorageClient, bucket_name: str, endpoint: str, file_path: str) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.file_path = file_path


class LocalMetaInfo:
    def __init__(self, client: StorageClient, dest_path: str) -> None:
        self.client = client
        self.dest_path = dest_path


def unpack_meta(meta):
    args = []
    for k, v in meta.__dict__.items():
        if k == "endpoint":
            continue
        args.append(v)
    return args


def compute_file_md5_by_chunk(file_name: str):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_boto3_meta(fp: str) -> Boto3MetaInfo:
    assert fp.startswith("s3://"), f"Path '{fp}' is not a boto3 url"
    parts = fp.lstrip("s3://").split(os.path.sep)
    match = boto3_url_re.match(parts[0])
    assert match is not None, f"url '{fp}' is not a valid boto3 url"
    bucket_name, endpoint = match.group(1), match.group(2)
    endpoint = "http://" + endpoint + ":80"
    return Boto3MetaInfo(None, bucket_name, endpoint, os.path.sep.join(parts[1:]))


def get_local_meta(fp: str) -> LocalMetaInfo:
    assert not fp.startswith("s3://"), f"Path '{fp}' is not a local path"
    return LocalMetaInfo(None, fp)


class Boto3Client(StorageClient):
    """
    Boto3Client
    """

    def __init__(
        self,
        s3_endpoint_url: str,
        use_threads: int = True,
        multipart_chunksize=8 * MB,
        max_concurrency: int = 10,
        multipart_threshold=100 * MB,
    ) -> None:
        """S3 object/file storage management class

        Args:
            s3_access_keys_id (str): S3 access key ID.
            s3_secret_access_key (str): S3 secret access key.
            use_threads (bool, optional): Whether to enable multipart. Defaults to True.
            multipart_chunksize (_type_, optional): Defaults to 8*MB.
            max_concurrency (int, optional): Defaults to 10.

        Raises:
            RuntimeError: Connection failures caused by misconfiguration or network problems.
        """
        super().__init__(boto3)
        self.botocore = botocore
        try:
            s3_access_key_id = os.environ["S3_ACCESS_KEY_ID"]
            s3_secret_access_key = os.environ["S3_SECRET_ACCESS_KEY_ID"]
        except KeyError as exc:
            raise RuntimeError(
                "Please set boto3 bucket 'S3_ACCESS_KEY_ID' and 'S3_SECRET_ACCESS_KEY_ID' using environment variable!"
            ) from exc

        self.client = self.handler.client(
            "s3",
            "",
            use_ssl=False,
            verify=False,
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )

        self.config = self.handler.s3.transfer.TransferConfig(
            multipart_threshold=multipart_threshold,
            max_concurrency=max_concurrency,
            multipart_chunksize=multipart_chunksize,
            use_threads=use_threads,
        )

    @staticmethod
    def sync_upload_fileobj(handler, bucket_name: str, fp: str, *args, saved_obj=None, **kwargs):
        assert saved_obj is not None, "saved_obj is None!"
        try:
            with io.BytesIO() as f:
                torch.save(saved_obj, f, *args, **kwargs)
                f.seek(0)
                handler.client.upload_fileobj(f, bucket_name, fp, Config=handler.config)
        except handler.botocore.exceptions.EndpointConnectionError as exc:
            raise RuntimeError(
                f"Boto3 Network Error: Please Check your Internet Connection in {socket.gethostname()}"
            ) from exc

    @staticmethod
    def load(handler, bucket_name: str, fp: str, *args, map_location="cpu", **kwargs) -> Dict:
        """
        Args:
            fp (str): Path to save, eg. s3://opennlplab/model_weights/xxx/ddd.pt
        """
        try:
            with io.BytesIO() as f:
                handler.client.download_fileobj(bucket_name, fp, f, Config=handler.config)
                f.seek(0)
                states = torch.load(f, *args, map_location=map_location, **kwargs)
        except handler.botocore.exceptions.EndpointConnectionError as exc:
            raise RuntimeError(
                f"Boto3 Network Error: Please Check your Internet Connection in {socket.gethostname()}"
            ) from exc
        return states

    @staticmethod
    def assert_fp_exists(
        handler,
        bucket_name: str,
        fp: str,
    ):
        assert len(list(handler.client.list_objects(Bucket=bucket_name, Prefix=fp)["Contents"])) > 0, fp

    @staticmethod
    def get_fns(handler, bucket_name: str, fp: str):
        """
        Ref: https://stackoverflow.com/questions/54314563/
        how-to-get-more-than-1000-objects-from-s3-by-using-list-objects-v2
        """
        paginator = handler.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=fp)

        folder_name_list = []
        for page in pages:
            for obj in page["Contents"]:
                fp: str = obj["Key"]
                folder_name_list.append(fp.rsplit("/", maxsplit=1)[1])
        return folder_name_list


class LocalClient(StorageClient):
    """
    Storage Client for local NFS.
    """

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=W0613
        super().__init__(None)

    @staticmethod
    def sync_upload_fileobj(handler, fp: str, *args, saved_obj=None, **kwargs):
        assert isinstance(handler, LocalClient)
        assert saved_obj is not None
        fp_dirname = os.path.dirname(fp)
        if not os.path.exists(fp_dirname):
            os.makedirs(fp_dirname, exist_ok=True)
        torch.save(saved_obj, fp, *args, **kwargs)

    @staticmethod
    def load(handler, fp: str, *args, map_location="cpu", **kwargs):
        assert isinstance(handler, LocalClient)
        assert os.path.exists(fp), f"{fp} is not found!"
        with open(fp, "rb") as f:
            states = torch.load(f, map_location=map_location, *args, **kwargs)
        return states

    @staticmethod
    def assert_fp_exists(handler, folder):
        assert isinstance(handler, LocalClient)
        assert os.path.exists(folder), folder

    @staticmethod
    def get_fns(handler, folder):
        assert isinstance(handler, LocalClient)
        assert os.path.exists(folder), f"folder '{folder}' not exists!"
        fns = os.listdir(folder)
        return fns

    @staticmethod
    def delete_obj(handler, fp: str):
        assert isinstance(handler, LocalClient)
        if not os.path.isdir(fp):
            os.remove(fp)


class StorageManager(metaclass=SingletonMeta):
    """
    Storage Manager for saving or loading checkpoint.
    """

    BACKEND_TYPE = {"boto3", "local"}
    BACKEND_INIT_METHOD = {
        "boto3": Boto3Client,
        "local": LocalClient,
    }
    CLI_DICT = {}

    def __init__(self) -> None:
        pass

    def _get_client(self, path=str) -> Union[Boto3MetaInfo, LocalMetaInfo]:
        """
        example:
        local:/path/to/checkpoint
        boto3:s3://model_weights/0331/120bi

        Args:
            path (str): _description_
        """
        try:
            backend, path = path.split(":", maxsplit=1)
        except Exception as exc:
            raise AttributeError(f"Given path '{path}' is not startwith backend prefix:'local/boto3'") from exc

        init_args = (None,)
        if backend == "local":
            meta_info = get_local_meta(path)
            backend_key = backend
        elif backend == "boto3":
            meta_info = get_boto3_meta(path)
            backend_key = backend + ":" + meta_info.endpoint
            init_args = (meta_info.endpoint,)
            if (
                "http_proxy" in os.environ
                or "https_proxy" in os.environ
                or "HTTP_PROXY" in os.environ
                or "HTTPS_PROXY" in os.environ
            ):
                raise RuntimeWarning(
                    "HTTP/HTTPS proxy is detected when using boto3, incorrectly setting \
the proxy may make boto3 unavailable or affect performance."
                )

        assert backend in StorageManager.BACKEND_TYPE, f"Unkown backend: {backend}"

        # boto3 backend need special treatment.
        if backend_key not in StorageManager.CLI_DICT:
            StorageManager.CLI_DICT.update({backend_key: StorageManager.BACKEND_INIT_METHOD[backend](*init_args)})

        meta_info.client = StorageManager.CLI_DICT[backend_key]

        return meta_info

    def assert_fp_exists(self, folder) -> None:
        meta = self._get_client(path=folder)
        meta.client.assert_fp_exists(*unpack_meta(meta))

    def get_fns(self, folder) -> List[str]:
        meta = self._get_client(path=folder)
        return meta.client.get_fns(*unpack_meta(meta))

    def save(self, save_path: str, saved_obj: Any, *args, **kwargs):
        meta = self._get_client(path=save_path)

        meta.client.sync_upload_fileobj(*unpack_meta(meta), *args, saved_obj=saved_obj, **kwargs)

    def load(self, load_path: str, *args, map_location="cpu", **kwargs) -> Any:

        meta = self._get_client(path=load_path)
        return meta.client.load(*unpack_meta(meta), map_location=map_location, *args, **kwargs)

    def delete_obj(self, fp: str):
        meta = self._get_client(path=fp)
        meta.client.delete_obj(*unpack_meta(meta))


storage_manager = StorageManager()
