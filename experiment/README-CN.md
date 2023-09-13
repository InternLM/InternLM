## 实验性环境镜像
本模块用于测试新版本环境，默认测试新环境 torch=2.0.1，flash-attention=2.1.0。新环境可能具有不稳定性，标准环境安装请参考：[安装文档](../doc/install.md)

### 镜像构建及拉取
构建镜像时请于 InternLM 根目录下执行 docker.Makefile，该文件与标准环境镜像共用，所使用的 Dockerfile 位于 experiment 目录下。也可直接从 https://hub.docker.com/r/internlm/internlm 拉取镜像，命令如下：
```bash
# 构建镜像
# ubuntu20.04
make -f docker.Makefile BASE_OS=ubuntu20.04 DOCKERFILE_PATH=./experiment/Dockerfile-ubuntu PYTORCH_VERSION=2.0.1 TORCHVISION_VERSION=0.15.2 TORCHAUDIO_VERSION=2.0.2 FLASH_ATTEN_VERSION=2.1.0
# centos7
make -f docker.Makefile BASE_OS=centos7 DOCKERFILE_PATH=./experiment/Dockerfile-centos PYTORCH_VERSION=2.0.1 TORCHVISION_VERSION=0.15.2 TORCHAUDIO_VERSION=2.0.2 FLASH_ATTEN_VERSION=2.1.0

# 拉取镜像
# ubuntu20.04
docker pull internlm/internlm:experiment-torch2.0.1-flashatten2.1.0-ubuntu20.04
# centos7
docker pull internlm/internlm:experiment-torch2.0.1-flashatten2.1.0-centos7
```

### 容器启动
对于使用 dockerfile 构建或拉取的本地标准镜像，使用如下命令启动并进入容器：
```bash
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 20g --network=host --name myinternlm internlm/internlm:experiment-torch2.0.1-flashatten2.1.0-centos7 bash
```
容器内默认目录即 `/InternLM`，根据[使用文档](../doc/usage.md)即可启动训练。