## 标准镜像配置及使用
dockerfile 的配置以及使用均通过 docker.Makefile 文件实现，执行如下命令即可 build 镜像：
``` 
make -f docker.Makefile
``` 
在 docker.Makefile 中可自定义基础镜像，环境版本等内容，对于 BASE_OS 分别支持 ubuntu 和 centos，在build命令中需要通过 -f 指定对应的标准 dockerfile。

## 标准镜像拉取
基于 ubuntu 和 centos 的标准镜像已经 build 完成也可直接拉取使用：

ubuntu:
```
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu18.04
```
centos:
```
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7
```

## 容器启动
对于使用 dockerfile 构建或拉取的本地标准镜像，使用如下命令启动并进入容器：
```
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 10g --network=host --name myinternlm li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu18.04 bash
```
容器内默认目录即 `/InternLM`，根据 README 即可启动训练。