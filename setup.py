import os
import re
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

pwd = os.path.dirname(__file__)

def readme():
    with open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    return content

def get_version():
    with open(os.path.join(pwd, 'version.txt'), 'r') as f:
        content = f.read()
    return content

class CustomInstall(install):
    def run(self):
        install.run(self)

        def custom_install_step(path, command):
            original_dir = os.getcwd()
            os.chdir(path)
            subprocess.check_call(command, shell=True)
            os.chdir(original_dir)

        custom_install_step('./requirements', 'pip install -r torch.txt')
        custom_install_step('./requirements', 'pip install -r runtime.txt')
        custom_install_step('./third_party/flash-attention', 'python setup.py install')
        custom_install_step('./third_party/flash-attention/csrc/fused_dense_lib', 'pip install -v .')
        custom_install_step('./third_party/flash-attention/csrc/xentropy', 'pip install -v .')
        custom_install_step('./third_party/flash-attention/csrc/rotary', 'pip install -v .')
        custom_install_step('./third_party/flash-attention/csrc/layer_norm', 'pip install -v .')
        custom_install_step('./third_party/apex', 'pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')

setup(
    name='InternLM',
    version=get_version(),
    description='an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    cmdclass={
        'install': CustomInstall,
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)
