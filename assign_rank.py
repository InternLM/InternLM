import os
import base64
import torch
import copy
import hashlib
import hmac
import re
import time
import uuid
from datetime import datetime
from subprocess import PIPE, STDOUT, Popen

import requests
import pandas as pd

class DLCJobInfo:
    """
    get dlc job detailed information
    """

    def __init__(
        self,
        endpoint,
        access_key_id,
        access_key_secret,
        protocol,
    ):
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.protocol = protocol

    def _get_canonicalized_resource(self, pathname, query):
        if len(query) <= 0:
            return pathname
        resource = f"{pathname}?"
        query_list = sorted(list(query))
        for key in query_list:
            if query[key] is not None:
                if query[key] == "":
                    s = f"{key}&"
                else:
                    value = self.to_string(query[key])
                    s = f"{key}={value}&"
            resource += s
        return resource[:-1]

    def to_string(self, s, encoding="utf-8"):
        if s is None:
            return s
        if isinstance(s, bytes):
            return s.decode(encoding)
        else:
            return str(s)

    def get_string_to_sign(self, method, url_path, headers, query):
        """使用请求信息生成待签名的字符串"""
        accept = "" if headers.get("accept") is None else headers.get("accept")
        content_md5 = "" if headers.get("content-md5") is None else headers.get("content-md5")
        content_type = "" if headers.get("content-type") is None else headers.get("content-type")
        date = "" if headers.get("date") is None else headers.get("date")
        header = f"{method}\n{accept}\n{content_md5}\n{content_type}\n{date}\n"

        canon_headers = self._get_canonicalized_headers(headers)
        canon_resource = self._get_canonicalized_resource(url_path, query)
        sign_str = header + canon_headers + canon_resource
        return sign_str

    def get_roasignature(self, string_to_sign, secret):
        """生成签名: 使用HMAC-256生成签名, 然后通过base64输出签名字符串。"""
        hash_val = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        signature = base64.b64encode(hash_val).decode("utf-8")
        return signature

    def _get_canonicalized_headers(self, headers):
        """将请求头排序后, 获取“x-acs-”作为前缀按字母序排序后拼接。
        注意, 按RFC2616, HTTP 请求头的名称是大小写不敏感的。
        """
        canon_keys = []
        for k in headers:
            if k.startswith("x-acs-"):
                canon_keys.append(k)
        canon_keys = sorted(canon_keys)
        canon_header = ""
        for k in canon_keys:
            canon_header += f"{k}:{ headers[k]}\n"
        return canon_header

    def do_request(self, api_product, api_query, api_method, api_path):
        """根据请求信息，生成认证信息，发送请求给到后端服务"""
        ts = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        signature_nonce = str(uuid.uuid4())
        headers = {
            "x-acs-signature-method": "HMAC-SHA1",
            "date": ts,
            "x-acs-signature-nonce": signature_nonce,
            "x-pai-product": api_product,
            "accept": "application/json",
        }
        api_url = f"{self.protocol}://{self.endpoint}{api_path}"
        # 使用请求信息，生成请求使用的签名(signature)，然后生成对应认证信息，在请求头里传递给到服务(authorization)
        string_to_sign = self.get_string_to_sign(method=api_method, url_path=api_path, headers=headers, query=api_query)
        signature = self.get_roasignature(string_to_sign=string_to_sign, secret=self.access_key_secret)
        headers["authorization"] = f"acs {self.access_key_id}:{signature}"
        resp = requests.request(
            method=api_method, url=api_url, params=api_query, headers=headers, verify=False, timeout=1
        )

        if resp.status_code != 200:
            print(resp.text)
            return None
        return resp.json()

    def do_get_job(self, jobid: str):
        # API请求的URL，以及参数
        api_path = f"/api/v1/jobs/{jobid}"
        # 填写请求query上的参数
        api_method = "GET"
        api_product = "dlc"
        api_query = {"jobid": jobid}

        job_meta = self.do_request(
            api_product=api_product, api_query=api_query, api_method=api_method, api_path=api_path
        )

        return job_meta

def get_sort_info(access_key_id, access_key_secret, endpoint, protocol):
    jobid = os.getenv("KUBERNETES_POD_NAME").split("-")[0]
    jobinfo = DLCJobInfo(access_key_id=access_key_id, access_key_secret=access_key_secret, endpoint=endpoint, protocol=protocol)
    pods = jobinfo.do_get_job(jobid)['Pods']
    node_id_dict = {}
    node_id_list = []
    for pod in pods:
        node_id_dict[pod['PodId']] = pod['NodeName']
        node_id_list.append(pod['NodeName'])
    
    return node_id_dict, node_id_list


def sort_node_id_list(node_id_list, node_switch_dict):
    # 自定义排序函数
    def custom_sort(node_id):
        return (node_switch_dict[node_id], node_id_list.index(node_id))

    # 对 node_list 进行排序
    sorted_node_id_list = sorted(node_id_list, key=custom_sort)
    
    return sorted_node_id_list


def get_info_for_rank_assign(access_key_id, access_key_secret, endpoint, protocol):
    print('begin2', flush=True)
    node_switch_dict = torch.load('node_switch_dict.pt')
    print('finish read_excel', flush=True)
    node_id_dict, node_id_list = get_sort_info(access_key_id, access_key_secret, endpoint, protocol)
    print('finish get_sort_info', flush=True)
    sorted_node_id_list = sort_node_id_list(node_id_list, node_switch_dict)
    print('finish sort_node_id_list', flush=True)
    
    return node_id_dict, sorted_node_id_list


if __name__ == "__main__":
    print('main', flush=True)
    access_key_id = 'lijiaxing'
    access_key_secret = 'cYMx8hgulMVD0RA0RSNp'
    endpoint = 'pai-proxy.cb210e3f99cd7403f8de2a630dcc99fc3.cn-wulanchabu.alicontainer.com:80'
    protocol = 'http'
    pod_id = os.getenv("KUBERNETES_POD_NAME")
    print('begin', flush=True)
    node_id_dict, sorted_node_id_list = get_info_for_rank_assign(access_key_id, access_key_secret, endpoint, protocol)
    currentu_node_id = node_id_dict[pod_id]
    node_index = sorted_node_id_list.index(currentu_node_id)
    print(f'currentu node id: {currentu_node_id}', flush=True)
    # print(f'local rank: {local_rank}', flush=True)
    print(f'node index: {node_index}', flush=True)
    # print(f'global rank: {node_index * 8 + local_rank}', flush=True)
    