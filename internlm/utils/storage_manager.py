#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import asyncio
import concurrent.futures
import hashlib
import io
import os
import pickle
import re
import socket
import stat
from asyncio import InvalidStateError
from asyncio.tasks import ALL_COMPLETED
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Union

import torch
import torch.distributed as dist

try:
    import boto3
    import botocore
except ImportError:
    pass

try:
    import tos
    from tos.utils import SizeAdapter
except ImportError:
    pass

try:
    import oss2
    from oss2 import SizedFileAdapter, determine_part_size
    from oss2.models import PartInfo
except ImportError:
    pass


class Logger:
    "Dummy logger"

    def info(self, mesage: str):
        print(f"Info: {mesage}", flush=True)

    def warning(self, mesage: str):
        print(f"Warning: {mesage}", flush=True)

    def error(self, mesage: str):
        print(f"Error: {mesage}", flush=True)


try:
    from internlm.utils.logger import get_logger

    logger = get_logger(__file__)
except ImportError:
    logger = Logger()


boto3_url_re = re.compile(r"([^\.]+)\.([\d\.]+)")
volc_url_re = re.compile(r"^(.*?)\.(.*)$")
ali_url_re = re.compile(r"([^/.]+)\.([^/.]+\..+)")

MB = 1024**2

storage_manager = None


def check_folder(fp: str):
    storage_manager.assert_fp_exists(fp)


def get_fns(fp: str):
    return storage_manager.get_fns(fp)


def llm_load(fp: str, **kwargs):
    return storage_manager.load(fp, **kwargs)


def llm_save(save_path: str, saved_obj: Any, **kwargs):
    storage_manager.save(save_path, to_save_obj=saved_obj, **kwargs)


def is_rank_for_log():
    if dist.is_initialized():
        return dist.get_rank() % 8 == 0
    return True


class StorageClient:
    """
    StorageClient as a client for s3 storage access.
    """

    def __init__(self, handler) -> None:
        self.handler = handler

    @staticmethod
    def load(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sync_upload_fileobj(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def async_upload_fileobj(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def assert_fp_exists(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_fns(*args, **kwargs):
        raise NotImplementedError


class Boto3MetaInfo:
    """Boto3 meta info for save/load etc."""

    def __init__(
        self,
        is_async,
        handler: StorageClient,
        bucket_name: str,
        endpoint: str,
        file_path: str,
        async_upload_fn: callable,
        local_nvme_path=None,
    ) -> None:
        # all need info.
        self.client = handler
        self.bucket_name = bucket_name
        self.file_path = file_path
        # only save need info.
        self.local_nvme_path = local_nvme_path
        self.is_async = is_async
        self.endpoint = endpoint
        self.async_upload_fn = async_upload_fn

    def __str__(self) -> str:
        return f"is_async: {self.is_async}, bucket_name:{self.bucket_name}, endpoint:{self.endpoint}, \
local_nvme_path: {self.local_nvme_path}"

    @staticmethod
    def unpack_boto3_save_meta(meta):
        if meta.is_async:
            return meta.client, meta.bucket_name, meta.file_path, meta.local_nvme_path
        else:
            return meta.client, meta.bucket_name, meta.file_path

    @staticmethod
    def unpack_boto3_nosave_meta(meta):
        return meta.client, meta.bucket_name, meta.file_path


class VolcMetaInfo:
    """Volc meta info for save/load etc."""

    def __init__(
        self,
        is_async,
        handler: StorageClient,
        bucket_name: str,
        endpoint: str,
        region: str,
        file_path: str,
        async_upload_fn: callable,
        local_nvme_path=None,
    ) -> None:
        # all need info.
        self.client = handler
        self.bucket_name = bucket_name
        self.file_path = file_path
        # only save need info.
        self.local_nvme_path = local_nvme_path
        self.is_async = is_async
        self.endpoint = endpoint
        self.region = region
        self.async_upload_fn = async_upload_fn

    def __str__(self) -> str:
        return f"is_async: {self.is_async}, bucket_name:{self.bucket_name}, endpoint:{self.endpoint}, \
region:{self.region}, local_nvme_path: {self.local_nvme_path}"

    @staticmethod
    def unpack_volc_save_meta(meta):
        if meta.is_async:
            return meta.client, meta.bucket_name, meta.file_path, meta.local_nvme_path
        else:
            return meta.client, meta.bucket_name, meta.file_path

    @staticmethod
    def unpack_volc_nosave_meta(meta):
        return meta.client, meta.bucket_name, meta.file_path


class AliMetaInfo:
    """Ali meta info for save/load etc."""

    def __init__(
        self,
        is_async,
        handler: StorageClient,
        bucket_name: str,
        endpoint: str,
        file_path: str,
        async_upload_fn: callable,
        local_nvme_path=None,
    ) -> None:
        # all need info.
        self.client = handler
        self.bucket_name = bucket_name
        self.file_path = file_path
        # only save need info.
        self.local_nvme_path = local_nvme_path
        self.is_async = is_async
        self.endpoint = endpoint
        self.async_upload_fn = async_upload_fn

    def __str__(self) -> str:
        return f"is_async: {self.is_async}, bucket_name:{self.bucket_name}, endpoint:{self.endpoint}, \
local_nvme_path: {self.local_nvme_path}"

    @staticmethod
    def unpack_ali_save_meta(meta):
        if meta.is_async:
            return meta.client, meta.file_path, meta.local_nvme_path
        else:
            return meta.client, meta.file_path

    @staticmethod
    def unpack_ali_nosave_meta(meta):
        return meta.client, meta.file_path


class LocalMetaInfo:
    """Local meta info for save/load etc."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.async_upload_fn = None
        self.is_async = False

    @staticmethod
    def unpack_local_save_meta(meta):
        return (meta.file_path,)

    @staticmethod
    def unpack_local_nosave_meta(meta):
        return (meta.file_path,)


def unpack_save_meta(meta: Union[Boto3MetaInfo, VolcMetaInfo, AliMetaInfo, LocalMetaInfo]):
    if isinstance(meta, Boto3MetaInfo):
        return Boto3MetaInfo.unpack_boto3_save_meta(meta)
    elif isinstance(meta, VolcMetaInfo):
        return VolcMetaInfo.unpack_volc_save_meta(meta)
    elif isinstance(meta, AliMetaInfo):
        return AliMetaInfo.unpack_ali_save_meta(meta)
    elif isinstance(meta, LocalMetaInfo):
        return LocalMetaInfo.unpack_local_save_meta(meta)
    else:
        raise ValueError(f"unkonwn meta info: {type(meta)}")


def unpack_nosave_meta(meta: Union[Boto3MetaInfo, VolcMetaInfo, AliMetaInfo, LocalMetaInfo]):
    if isinstance(meta, Boto3MetaInfo):
        return Boto3MetaInfo.unpack_boto3_nosave_meta(meta)
    elif isinstance(meta, VolcMetaInfo):
        return VolcMetaInfo.unpack_volc_nosave_meta(meta)
    elif isinstance(meta, AliMetaInfo):
        return AliMetaInfo.unpack_ali_nosave_meta(meta)
    elif isinstance(meta, LocalMetaInfo):
        return LocalMetaInfo.unpack_local_nosave_meta(meta)
    else:
        raise ValueError(f"unkonwn meta info: {type(meta)}")


def compute_file_md5_by_chunk(file_name: str):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def try_get_storage_backend(path: str):
    if path.startswith("s3:"):
        if is_rank_for_log():
            logger.warning(f"path: '{path}' not start with backend prefix, guess it is the backend of boto3.")
        return "boto3", path
    elif path.startswith("vc:"):
        if is_rank_for_log():
            logger.warning(f"path: '{path}' not start with backend prefix, guess it is the backend of volc.")
        return "volc", path
    elif path.startswith("ali:"):
        if is_rank_for_log():
            logger.warning(f"path: '{path}' not start with backend prefix, guess it is the backend of ali.")
        return "oss2", path
    else:
        sre = path.split(":", maxsplit=1)
        if len(sre) == 1:
            if is_rank_for_log():
                logger.warning(f"path: '{path}' not start with backend prefix, guess it is the backend of local.")
            return "local", sre[0]
        else:
            return sre[0], sre[1]  # (backend_prefix, splited_path)


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
            ACCESS_KEY (str): S3 access key ID.
            SECRET_ACCESS_KEY (str): S3 secret access key.
            use_threads (bool, optional): Whether to enable multipart. Defaults to True.
            multipart_chunksize (_type_, optional): Defaults to 8*MB.
            max_concurrency (int, optional): Defaults to 10.

        Raises:
            RuntimeError: Connection failures caused by misconfiguration or network problems.
        """
        super().__init__(boto3)
        self.botocore = botocore
        try:
            if os.environ.get("S3_ACCESS_KEY_ID") is not None and os.environ.get("ACCESS_KEY") is not None:
                s3_access_key_id = os.environ["ACCESS_KEY"]
                logger.warning("Both 'S3_ACCESS_KEY_ID' and 'ACCESS_KEY' exist, 'ACCESS_KEY' will be used by default")
            elif os.environ.get("ACCESS_KEY") is None:
                s3_access_key_id = os.environ["S3_ACCESS_KEY_ID"]
            else:
                s3_access_key_id = os.environ["ACCESS_KEY"]

            if (
                os.environ.get("S3_SECRET_ACCESS_KEY_ID") is not None
                and os.environ.get("SECRET_ACCESS_KEY") is not None
            ):
                s3_secret_access_key = os.environ["SECRET_ACCESS_KEY"]
                logger.warning(
                    "Both 'S3_SECRET_ACCESS_KEY_ID' and 'SECRET_ACCESS_KEY' exist, "
                    "'SECRET_ACCESS_KEY' will be used by default"
                )
            elif os.environ.get("SECRET_ACCESS_KEY") is None:
                s3_secret_access_key = os.environ["S3_SECRET_ACCESS_KEY_ID"]
            else:
                s3_secret_access_key = os.environ["SECRET_ACCESS_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                "Please set boto3 bucket 'ACCESS_KEY' and 'SECRET_ACCESS_KEY' using environment variable!"
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
    def sync_upload_fileobj(handler, bucket_name: str, fp: str, saved_obj=None, **kwargs):
        assert saved_obj is not None, "saved_obj is None!"
        try:
            with io.BytesIO() as f:
                torch.save(saved_obj, f, **kwargs)
                f.seek(0)
                handler.client.upload_fileobj(f, bucket_name, fp, Config=handler.config)
        except handler.botocore.exceptions.EndpointConnectionError as exc:
            raise RuntimeError(
                f"Boto3 Network Error: Please Check your Internet Connection in {socket.gethostname()}"
            ) from exc

    @staticmethod
    def load(handler, bucket_name: str, fp: str, **kwargs) -> Dict:
        """
        Args:
            fp (str): Path to save, eg. s3://opennlplab/model_weights/xxx/ddd.pt
        """
        try:
            with io.BytesIO() as f:
                handler.client.download_fileobj(bucket_name, fp, f, Config=handler.config)
                f.seek(0)
                states = torch.load(f, **kwargs)
        except handler.botocore.exceptions.EndpointConnectionError as exc:
            raise RuntimeError(
                f"Boto3 Network Error: Please Check your Internet Connection in {socket.gethostname()}"
            ) from exc
        return states

    @staticmethod
    def assert_fp_exists(handler, bucket_name: str, fp: str):  # pylint: disable=W0613
        assert len(list(handler.client.list_objects(Bucket=bucket_name, Prefix=fp)["Contents"])) > 0, fp

    @staticmethod
    def is_fp_exists(handler, bucket_name: str, fp: str):  # pylint: disable=W0613
        re = handler.client.list_objects(Bucket=bucket_name, Prefix=fp)
        if "Contents" in re:
            return len(list(re["Contents"])) > 0
        else:
            return False

    @staticmethod
    def get_fns(handler, bucket_name: str, fp: str):
        """
        Ref: https://stackoverflow.com/questions/54314563/
        how-to-get-more-than-1000-objects-from-s3-by-using-list-objects-v2
        """
        if Boto3Client.is_fp_exists(handler, bucket_name, fp):
            paginator = handler.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=fp)
            folder_name_list = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        pth: str = obj["Key"]
                        folder_name_list.append(pth.split(fp, maxsplit=1)[1].strip("/").split("/", maxsplit=1)[0])
            return list(set(folder_name_list))
        else:
            if is_rank_for_log():
                logger.warning(f"'{fp}' not found!")
            return None

    @staticmethod
    def async_upload_fileobj(handler, bucket_name: str, fp: str, local_nvme_path: str):
        try:
            with open(local_nvme_path, "rb") as f:
                handler.client.upload_fileobj(f, bucket_name, fp, Config=handler.config)
        except handler.botocore.exceptions.EndpointConnectionError as exc:
            raise RuntimeError(
                f"Boto3 Network Error: Please Check your Internet Connection in {socket.gethostname()}"
            ) from exc
        except Exception as e:
            raise e

    @staticmethod
    def delete_obj(handler, fp: str):
        raise NotImplementedError("boto3 not support delete_obj")


class VolcClient(StorageClient):
    """
    VolcClient
    """

    def __init__(
        self,
        endpoint: str,
        region: str,
    ) -> None:
        """Volc object/file storage management class

        Args:
            ACCESS_KEY (str): Volc access key ID.
            SECRET_ACCESS_KEY (str): Volc secret access key.
            endpoint (str): Volc tos endpoint.
            region (str): Volc tos region.

        """
        super().__init__(tos)

        try:
            if os.environ.get("VOLC_ACCESS_KEY_ID") is not None and os.environ.get("ACCESS_KEY") is not None:
                access_key = os.environ["ACCESS_KEY"]
                logger.warning("Both 'VOLC_ACCESS_KEY_ID' and 'ACCESS_KEY' exist, 'ACCESS_KEY' will be used by default")
            elif os.environ.get("ACCESS_KEY") is None:
                access_key = os.environ["VOLC_ACCESS_KEY_ID"]
            else:
                access_key = os.environ["ACCESS_KEY"]

            if (
                os.environ.get("VOLC_SECRET_ACCESS_KEY_ID") is not None
                and os.environ.get("SECRET_ACCESS_KEY") is not None
            ):
                secret_key = os.environ["SECRET_ACCESS_KEY"]
                logger.warning(
                    "Both 'VOLC_SECRET_ACCESS_KEY_ID' and 'SECRET_ACCESS_KEY' exist, "
                    "'SECRET_ACCESS_KEY' will be used by default"
                )
            elif os.environ.get("SECRET_ACCESS_KEY") is None:
                secret_key = os.environ["VOLC_SECRET_ACCESS_KEY_ID"]
            else:
                secret_key = os.environ["SECRET_ACCESS_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                "Please set 'ACCESS_KEY' and 'SECRET_ACCESS_KEY'",
                "using environment variable!",
            ) from exc

        self.client = self.handler.TosClientV2(access_key, secret_key, endpoint, region, enable_crc=False)

    @staticmethod
    def sync_upload_fileobj(handler, bucket_name: str, fp: str, saved_obj=None, **kwargs):
        assert saved_obj is not None, "saved_obj is None!"
        try:
            with io.BytesIO() as f:
                torch.save(saved_obj, f, **kwargs)
                f.seek(0)
                handler.client.put_object(bucket_name, fp, content=f)
        except handler.handler.exceptions.TosClientError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with client error, message:{exc.message}, cause: {exc.cause}"
            ) from exc
        except handler.handler.exceptions.TosServerError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with server error, code: {exec.code}",
                f"error with request id: {exec.request_id}",
                f"error with message: {exec.message}",
                f"error with http code: {exec.status_code}",
            ) from exc

    @staticmethod
    def load(handler, bucket_name: str, fp: str, **kwargs) -> Dict:
        """
        Args:
            fp (str): Path to save, eg. vc://opennlplab/model_weights/xxx/ddd.pt
        """
        try:
            object_stream = handler.client.get_object(bucket_name, fp)
            buffer = io.BytesIO(object_stream.read())
            states = torch.load(buffer, **kwargs)
        except handler.handler.exceptions.TosClientError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with client error, message:{exc.message}, cause: {exc.cause}"
            ) from exc
        except handler.handler.exceptions.TosServerError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with server error, code: {exec.code}",
                f"error with request id: {exec.request_id}",
                f"error with message: {exec.message}",
                f"error with http code: {exec.status_code}",
            ) from exc

        return states

    @staticmethod
    def assert_fp_exists(handler, bucket_name: str, fp: str):  # pylint: disable=W0613
        assert len(list(handler.client.list_objects_type2(bucket_name, prefix=fp).contents)) > 0, fp

    @staticmethod
    def is_fp_exists(handler, bucket_name: str, fp: str):  # pylint: disable=W0613
        re = handler.client.list_objects_type2(bucket_name, prefix=fp)
        if hasattr(re, "contents"):
            return len(list(re.contents)) > 0
        else:
            return False

    @staticmethod
    def get_fns(handler, bucket_name: str, fp: str):
        if VolcClient.is_fp_exists(handler, bucket_name, fp):
            folder_name_list = []
            result = handler.client.list_objects_type2(bucket_name, prefix=fp)
            if hasattr(result, "contents"):
                for iterm in result.contents:
                    pth = iterm.key
                    folder_name_list.append(pth.split(fp, maxsplit=1)[1].strip("/").split("/", maxsplit=1)[0])

            while result.is_truncated:
                result = handler.client.list_objects_type2(
                    bucket_name, prefix=fp, continuation_token=result.next_continuation_token
                )
                if hasattr(result, "contents"):
                    for iterm in result.contents:
                        pth = iterm.key
                        folder_name_list.append(pth.split(fp, maxsplit=1)[1].strip("/").split("/", maxsplit=1)[0])

            return list(set(folder_name_list))

        else:
            if is_rank_for_log():
                logger.warning(f"'{fp}' not found!")
            return None

    @staticmethod
    def async_upload_fileobj(handler, bucket_name: str, fp: str, local_nvme_path: str):
        try:
            total_size = os.path.getsize(local_nvme_path)
            part_size = 5 * 1024 * 1024

            multi_result = handler.client.create_multipart_upload(bucket_name, fp)

            upload_id = multi_result.upload_id
            parts = []

            # Upload shard data
            with open(local_nvme_path, "rb") as f:
                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    out = handler.client.upload_part(
                        bucket_name,
                        fp,
                        upload_id,
                        part_number,
                        content=SizeAdapter(f, num_to_upload, init_offset=offset),
                    )
                    parts.append(out)
                    offset += num_to_upload
                    part_number += 1

            # Complete the multipart upload task
            handler.client.complete_multipart_upload(bucket_name, fp, upload_id, parts)

        except handler.handler.exceptions.TosClientError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with client error, message:{exc.message}, cause: {exc.cause}"
            ) from exc
        except handler.handler.exceptions.TosServerError as exc:
            raise RuntimeError(
                f"Volc Network Error: fail with server error, code: {exec.code}",
                f"error with request id: {exec.request_id}",
                f"error with message: {exec.message}",
                f"error with http code: {exec.status_code}",
                f"error with ec: {exec.ec}",
                f"error with request url: {exec.request_url}",
            ) from exc
        except Exception as e:
            raise e

    @staticmethod
    def delete_obj(handler, fp: str):
        raise NotImplementedError("volc not support delete_obj")


class AliClient(StorageClient):
    """
    AliClient
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint: str,
    ) -> None:
        """Ali object/file storage management class

        Args:
            ACCESS_KEY (str): Ali access key ID.s
            SECRET_ACCESS_KEY (str): Ali secret access key.
            endpoint (str): Ali tos endpoint.
            bucket_name (str): Ali tos bucket_name.

        """
        super().__init__(oss2)

        try:
            if os.environ.get("ALI_ACCESS_KEY_ID") is not None and os.environ.get("ACCESS_KEY") is not None:
                access_key = os.environ["ACCESS_KEY"]
                logger.warning("Both 'ALI_ACCESS_KEY_ID' and 'ACCESS_KEY' exist, 'ACCESS_KEY' will be used by default")
            elif os.environ.get("ACCESS_KEY") is None:
                access_key = os.environ["ALI_ACCESS_KEY_ID"]
            else:
                access_key = os.environ["ACCESS_KEY"]

            if (
                os.environ.get("ALI_SECRET_ACCESS_KEY_ID") is not None
                and os.environ.get("SECRET_ACCESS_KEY") is not None
            ):
                secret_key = os.environ["SECRET_ACCESS_KEY"]
                logger.warning(
                    "Both 'ALI_SECRET_ACCESS_KEY_ID' and 'SECRET_ACCESS_KEY' exist, "
                    "'SECRET_ACCESS_KEY' will be used by default"
                )
            elif os.environ.get("SECRET_ACCESS_KEY") is None:
                secret_key = os.environ["ALI_SECRET_ACCESS_KEY_ID"]
            else:
                secret_key = os.environ["SECRET_ACCESS_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                "Please set 'ACCESS_KEY' and 'SECRET_ACCESS_KEY'",
                "using environment variable!",
            ) from exc

        self.auth = self.handler.Auth(access_key, secret_key)
        self.client = self.handler.Bucket(self.auth, endpoint, bucket_name, enable_crc=False)

    @staticmethod
    def sync_upload_fileobj(handler, fp: str, saved_obj=None, **kwargs):
        assert saved_obj is not None, "saved_obj is None!"
        try:
            with io.BytesIO() as f:
                torch.save(saved_obj, f, **kwargs)
                f.seek(0)
                handler.client.put_object(fp, f)
        except Exception as e:
            raise e

    @staticmethod
    def load(handler, fp: str, **kwargs) -> Dict:
        """
        Args:
            fp (str): Path to save, eg. ali://opennlplab/model_weights/xxx/ddd.pt
        """
        try:
            object_stream = handler.client.get_object(fp)
            buffer = io.BytesIO(object_stream.read())
            states = torch.load(buffer, **kwargs)
        except Exception as e:
            raise e

        return states

    @staticmethod
    def assert_fp_exists(handler, fp: str):  # pylint: disable=W0613
        assert len(list(handler.handler.ObjectIteratorV2(handler.client, prefix=fp))) > 0, fp

    @staticmethod
    def is_fp_exists(handler, fp: str):  # pylint: disable=W0613
        return len(list(handler.handler.ObjectIteratorV2(handler.client, prefix=fp))) > 0

    @staticmethod
    def get_fns(handler, fp: str):
        if AliClient.is_fp_exists(handler, fp):
            folder_name_list = []
            for obj in handler.handler.ObjectIteratorV2(handler.client, prefix=fp):
                folder_name_list.append(obj.key.split(fp, maxsplit=1)[1].strip("/").split("/", maxsplit=1)[0])

            return list(set(folder_name_list))
        else:
            if is_rank_for_log():
                logger.warning(f"'{fp}' not found!")
            return None

    @staticmethod
    def async_upload_fileobj(handler, fp: str, local_nvme_path: str):
        try:
            total_size = os.path.getsize(local_nvme_path)
            part_size = determine_part_size(total_size, preferred_size=5 * 1024 * 1024)
            upload_id = handler.client.init_multipart_upload(fp).upload_id
            parts = []
            with open(local_nvme_path, "rb") as fileobj:
                part_number = 1
                offset = 0
                while offset < total_size:
                    num_to_upload = min(part_size, total_size - offset)
                    # Calling the SizedFileAdapter method will generate a new file object
                    # and recalculate the starting append position.
                    result = handler.client.upload_part(
                        fp, upload_id, part_number, SizedFileAdapter(fileobj, num_to_upload)
                    )
                    parts.append(PartInfo(part_number, result.etag))

                    offset += num_to_upload
                    part_number += 1

            headers = dict()
            handler.client.complete_multipart_upload(fp, upload_id, parts, headers=headers)
        except Exception as e:
            raise e

    @staticmethod
    def delete_obj(handler, fp: str):
        raise NotImplementedError("ali not support delete_obj")


class LocalClient(StorageClient):
    """
    Storage Client for local NFS.
    """

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=W0613
        super().__init__(None)

    @staticmethod
    def sync_upload_fileobj(fp: str, saved_obj=None, **kwargs):
        assert saved_obj is not None
        fp_dirname = os.path.dirname(fp)
        try:
            if not os.path.exists(fp_dirname):
                os.makedirs(fp_dirname, exist_ok=True)
        except FileNotFoundError:
            pass
        torch.save(saved_obj, fp, **kwargs)

    @staticmethod
    def load(load_path: str, **kwargs):
        assert os.path.exists(load_path), f"{load_path} is not found!"
        with open(load_path, "rb") as f:
            states = torch.load(f, **kwargs)
        return states

    @staticmethod
    def assert_fp_exists(folder):
        assert os.path.exists(folder), folder

    @staticmethod
    def get_fns(folder):
        if not os.path.exists(folder):
            if is_rank_for_log():
                logger.warning(f"'{folder}' not found!")
            return None
        else:
            return os.listdir(folder)

    @staticmethod
    def delete_obj(fp: str):
        if not os.path.isdir(fp):
            os.remove(fp)


def get_tmp_file_name(tmp_local_folder: str, fp: str):
    """
    It should be noted that all our temporary files will be stored in the same folder,
    so the file name passed upstream must be unique.
    """
    base_path = os.path.join(tmp_local_folder, fp.split("/")[-1])
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    pid = os.getpid()
    # step = self.step_counter
    return "-".join([base_path, current_time, str(pid)]) + ".tmpfile"  # , str(step)


def get_boto3_meta(fp: str, tmp_local_folder: str, is_async: bool) -> Boto3MetaInfo:
    assert fp.startswith("s3://"), f"Path '{fp}' is not a boto3 url"
    parts = fp.lstrip("s3://").split(os.path.sep)
    match = boto3_url_re.match(parts[0])
    assert match is not None, f"url '{fp}' is not a valid boto3 url"
    bucket_name, endpoint = match.group(1), match.group(2)
    endpoint = "http://" + endpoint + ":80"
    if is_async:
        tmp_step_file = get_tmp_file_name(tmp_local_folder, fp)
    else:
        tmp_step_file = None
    return Boto3MetaInfo(
        is_async=is_async,
        handler=None,
        bucket_name=bucket_name,
        endpoint=endpoint,
        file_path=os.path.sep.join(parts[1:]),
        async_upload_fn=Boto3Client.async_upload_fileobj,
        local_nvme_path=tmp_step_file,
    )


def get_volc_meta(fp: str, tmp_local_folder: str, is_async: bool) -> VolcMetaInfo:
    assert fp.startswith("vc://"), f"Path '{fp}' is not a volc url"
    parts = fp.lstrip("vc://").split(os.path.sep)
    match = volc_url_re.match(parts[0])
    assert match is not None, f"url '{fp}' is not a valid volc url"
    bucket_name, endpoint = match.group(1), match.group(2)
    region = endpoint.split(".")
    region = region[0].split("-")
    region = "-".join(region[1:])

    if is_async:
        tmp_step_file = get_tmp_file_name(tmp_local_folder, fp)
    else:
        tmp_step_file = None
    return VolcMetaInfo(
        is_async=is_async,
        handler=None,
        bucket_name=bucket_name,
        endpoint=endpoint,
        region=region,
        file_path=os.path.sep.join(parts[1:]),
        async_upload_fn=VolcClient.async_upload_fileobj,
        local_nvme_path=tmp_step_file,
    )


def get_ali_meta(fp: str, tmp_local_folder: str, is_async: bool) -> AliMetaInfo:
    assert fp.startswith("ali://"), f"Path '{fp}' is not a ali url"
    parts = fp.lstrip("ali://").split(os.path.sep)
    match = ali_url_re.match(parts[0])
    assert match is not None, f"url '{fp}' is not a valid ali url"
    bucket_name, endpoint = match.group(1), match.group(2)

    if is_async:
        tmp_step_file = get_tmp_file_name(tmp_local_folder, fp)
    else:
        tmp_step_file = None
    return AliMetaInfo(
        is_async=is_async,
        handler=None,
        bucket_name=bucket_name,
        endpoint=endpoint,
        file_path=os.path.sep.join(parts[1:]),
        async_upload_fn=AliClient.async_upload_fileobj,
        local_nvme_path=tmp_step_file,
    )


def get_local_meta(fp: str) -> LocalMetaInfo:
    assert (
        not fp.startswith("s3://") and not fp.startswith("vc://") and not fp.startswith("ali://")
    ), f"Path '{fp}' is not a local path"
    return LocalMetaInfo(fp)


def get_mount_point_free_size(path: str):
    """
        Returns the remaining space of the temporary storage mount point as a percentage.
    Args:
        path (str): temporary storage folder path.

    Raises:
        FileNotFoundError: If the temporary storage folder does not exist,
        an error will be reported。
    """
    if os.path.exists(path):
        st = os.statvfs(path)
        # f_bavail: Number of free blocks for unprivileged users.
        # f_bsize: Filesystem block size.
        # return unit is TB.
        return st.f_bavail * st.f_bsize / (1024**3)


def check_tmp_folder_accessibility(tmp_local_folder: str):
    """
    Check access permissions for temporary storage.
    """
    ret = True
    if os.path.exists(tmp_local_folder):
        ret &= os.access(tmp_local_folder, os.W_OK)
        ret &= os.access(tmp_local_folder, os.R_OK)
        if ret is False:
            error_str = f'{socket.gethostname()} dose not have read and write permissions on {tmp_local_folder}"'
            raise RuntimeError(error_str)


class SingletonMeta(type):
    """
    Singleton Meta.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and a instance has been created."
        return cls._instances[cls]


class StorageManager(metaclass=SingletonMeta):
    """
    Storage Manager for saving or loading checkpoint.
    TODO: add a thread to poll the asynchronous storage state.
    """

    BACKEND_TYPE = {"boto3", "local", "volc", "oss2"}
    BACKEND_INIT_METHOD = {
        "boto3": Boto3Client,
        "local": LocalClient,
        "volc": VolcClient,
        "oss2": AliClient,
    }
    CLI_DICT = {}

    def __init__(self, enable_save, tmp_local_folder="/dev/shm/test/", async_mode=True, n_async_workers=8) -> None:
        self._exception_list = []
        self._to_be_del_files = []
        self._async_stack = []
        self.upload_count = 0
        self.tmp_local_folder = tmp_local_folder
        self.async_mode = async_mode
        self.has_warning = False
        self._async_loop = None
        self._thread_pool = None
        self.latest_save_folder = None
        self.latest_save_step = 0
        self.async_task_peeding = False

        if enable_save and self.async_mode:
            self._async_loop = asyncio.new_event_loop()
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_async_workers)

            check_tmp_folder_accessibility(os.path.dirname(self.tmp_local_folder))

            # Try to create tmp folder
            try:
                os.makedirs(self.tmp_local_folder, exist_ok=True)
                os.chmod(self.tmp_local_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except FileExistsError:
                pass

            # In case it is a directory created by other users, we check the permissions again.
            check_tmp_folder_accessibility(self.tmp_local_folder)

            # Try to clean tmp folder's empty folder.
            self.try_delete_tmpfile(self.tmp_local_folder)

            # Avaliable storeage space check.
            free_size = get_mount_point_free_size(self.tmp_local_folder)
            if free_size < 0.1:
                logger.error(f'tmp_local_folder only have "{free_size}" GB free space, less then 100 GB!')
                raise RuntimeError(f"Insufficient temporary storage space on {socket.gethostname()}")

    def _get_client(
        self, path: str, async_mode: bool = False
    ) -> Union[Boto3MetaInfo, VolcMetaInfo, AliMetaInfo, LocalMetaInfo]:
        """
        example:
        local:/path/to/checkpoint
        boto3:s3://model_weights/0331/120bi
        volc:vc://model_weights/0331/120bi
        oss2:ali://model_weights/0331/120bi

        Args:
            path (str): _description_
        """
        backend, path = try_get_storage_backend(path)

        init_args = (None,)
        if backend == "local":
            meta_info = get_local_meta(path)
            backend_key = backend
        elif backend == "boto3":
            meta_info = get_boto3_meta(path, self.tmp_local_folder, async_mode)
            backend_key = backend + ":" + meta_info.endpoint
            init_args = (meta_info.endpoint,)
            if (
                "http_proxy" in os.environ
                or "https_proxy" in os.environ
                or "HTTP_PROXY" in os.environ
                or "HTTPS_PROXY" in os.environ
            ):
                if not self.has_warning and is_rank_for_log():
                    logger.warning(
                        "HTTP/HTTPS proxy is detected when using boto3, incorrectly setting \
    the proxy may make boto3 unavailable or affect performance."
                    )
                    self.has_warning = True
        elif backend == "volc":
            meta_info = get_volc_meta(path, self.tmp_local_folder, async_mode)
            backend_key = backend + ":" + meta_info.endpoint
            init_args = (
                meta_info.endpoint,
                meta_info.region,
            )
            if (
                "http_proxy" in os.environ
                or "https_proxy" in os.environ
                or "HTTP_PROXY" in os.environ
                or "HTTPS_PROXY" in os.environ
            ):
                if not self.has_warning and is_rank_for_log():
                    logger.warning(
                        "HTTP/HTTPS proxy is detected when using volc, incorrectly setting \
    the proxy may make volc unavailable or affect performance."
                    )
                    self.has_warning = True
        elif backend == "oss2":
            meta_info = get_ali_meta(path, self.tmp_local_folder, async_mode)
            backend_key = backend + ":" + meta_info.endpoint
            init_args = (
                meta_info.bucket_name,
                meta_info.endpoint,
            )
            if (
                "http_proxy" in os.environ
                or "https_proxy" in os.environ
                or "HTTP_PROXY" in os.environ
                or "HTTPS_PROXY" in os.environ
            ):
                if not self.has_warning and is_rank_for_log():
                    logger.warning(
                        "HTTP/HTTPS proxy is detected when using oss2, incorrectly setting \
    the proxy may make oss2 unavailable or affect performance."
                    )
                    self.has_warning = True

        assert backend in StorageManager.BACKEND_TYPE, f"Unkown backend: {backend}"

        # boto3, volc and oss2 backend need special treatment.
        if backend_key not in StorageManager.CLI_DICT:
            StorageManager.CLI_DICT.update({backend_key: StorageManager.BACKEND_INIT_METHOD[backend](*init_args)})

        meta_info.client = StorageManager.CLI_DICT[backend_key]

        return meta_info

    def assert_fp_exists(self, folder) -> None:
        meta = self._get_client(path=folder)
        meta.client.assert_fp_exists(*unpack_nosave_meta(meta))

    def get_fns(self, folder) -> List[str]:
        meta = self._get_client(path=folder)
        return meta.client.get_fns(*unpack_nosave_meta(meta))

    def save(self, save_path: str, to_save_obj: Any, async_upload=None, **kwargs):
        if async_upload is None:
            async_upload = self.async_mode

        if (
            not save_path.startswith("boto3:")
            and not save_path.startswith("volc:")
            and not save_path.startswith("oss2:")
        ):
            async_upload = False

        meta = self._get_client(save_path, async_upload)

        if async_upload:
            assert (
                self.tmp_local_folder
            ), "StorageManager is not setted tmp_local_folder, so async save cannot be performed."
            tmp_step_file = meta.local_nvme_path
            self._to_be_del_files.append(tmp_step_file)
            with open(tmp_step_file, "wb") as f:
                torch.save(to_save_obj, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            self.async_executor(meta.async_upload_fn, *unpack_save_meta(meta))
            os.chmod(tmp_step_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            self.async_task_peeding = True
        else:
            meta.client.sync_upload_fileobj(*unpack_save_meta(meta), saved_obj=to_save_obj, **kwargs)
            self.upload_count += 1

    def load(self, load_path: str, **kwargs) -> Any:
        self.wait()
        meta = self._get_client(path=load_path)

        return meta.client.load(*unpack_nosave_meta(meta), **kwargs)

    def delete_obj(self, fp: str):
        meta = self._get_client(path=fp)
        meta.client.delete_obj(*unpack_nosave_meta(meta))

    def _del_tmp_folder(self):
        for fp in self._to_be_del_files:
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass
            except SystemError as e:
                logger.error(f'delete file: {fp}, failed for reason:"{e}"')
            else:
                pass

    def try_delete_tmpfile(self, tmp_dir: str):
        """Delete temporary files in tmp_dir."""

        for filename in os.listdir(tmp_dir):
            if filename.endswith(".tmpfile"):
                file_path = os.path.join(tmp_dir, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Delete tmpfile: {file_path}")
                except OSError:
                    # Ignore deletion errors
                    pass

    async def _sync_tasks(self) -> Awaitable[None]:
        if self._async_stack:
            await asyncio.wait(self._async_stack, return_when=ALL_COMPLETED)
            count = 0
            while self._async_stack:
                t = self._async_stack[0]
                try:
                    e = t.exception()
                    if e:
                        self._exception_list.append((e, count))
                        logger.error(f"File:{self._to_be_del_files[count]}, upload failed for {e}")
                        # raise e
                    count += 1
                    self._async_stack.pop(0)
                except InvalidStateError:
                    # Not finished. https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.exception
                    pass

    def async_executor(self, fn: Callable, *args, **kwargs) -> None:
        """
        Overview:
            Execute task in background, then apppend the future instance in _async_stack.
        Arguments:
            - fn (:obj:`Callable`): Synchronization fuction.
        """
        if not self._async_loop:
            raise RuntimeError("Event loop was not initialized, please call this function in async or parallel mode")
        t = self._async_loop.run_in_executor(self._thread_pool, fn, *args, **kwargs)
        self._async_stack.append(t)

    def wait(self) -> bool:
        """Wait for async operations to complete."""

        if not self.async_mode:
            return

        if not self.async_task_peeding:
            return

        if self._async_loop:
            self._async_loop.run_until_complete(self._sync_tasks())

        if self._exception_list:
            for error_msg, file_id in self._exception_list:
                logger.error(
                    f"Node:{socket.gethostname()}, Error: Checkpoint {self._to_be_del_files[file_id]} "
                    f"failed on step {self.upload_count}: {error_msg}"
                )

                # TODO: Re-upload in sync mode
                raise RuntimeError(
                    f"Failed to upload {self._to_be_del_files[file_id]} " f"on step {self.upload_count}: {error_msg}"
                )

        self._del_tmp_folder()
        self._exception_list.clear()
        self._to_be_del_files.clear()
        self.async_task_peeding = False

        if is_rank_for_log():
            self.upload_count += 1
            if self.async_mode and self.latest_save_folder:
                self.save(
                    os.path.join(self.latest_save_folder, f"{self.latest_save_step}.step"),
                    to_save_obj=dict({"step": self.latest_save_step}),
                    async_upload=False,
                )
                self.latest_save_folder = None


storage_manager: StorageManager = None


def init_storage_manager(enable_save_ckpt, async_upload_tmp_folder, async_upload):
    global storage_manager
    storage_manager = StorageManager(
        enable_save_ckpt,
        tmp_local_folder=async_upload_tmp_folder,
        async_mode=async_upload,
    )


def get_storage_manager():
    assert storage_manager is not None, "storage_manager has not been init!"
    return storage_manager


def wait_async_upload_finish():
    dist.barrier()
    storage_manager.wait()
