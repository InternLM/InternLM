## 环境安装

### 环境准备
首先，需要安装的依赖包及对应版本列表如下：
- Python == 3.10
- GCC == 10.2.0
- MPFR == 4.1.0
- CUDA >= 11.7
- Pytorch >= 1.13.1
- Transformers >= 4.28.0
- Flash-Attention >= v1.0.5
- Apex == 23.05
- Ampere或者Hopper架构的GPU (例如H100, A100)
- Linux OS

以上依赖包安装完成后，需要更新配置系统环境变量：
```bash
export CUDA_PATH={path_of_cuda_11.7}
export GCC_HOME={path_of_gcc_10.2.0}
export MPFR_HOME={path_of_mpfr_4.1.0}
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${MPFR_HOME}/lib:${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
export PATH=${GCC_HOME}/bin:${CUDA_PATH}/bin:$PATH
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++
```

### 环境安装
将项目`internlm`及其依赖子模块，从 github 仓库中 clone 下来，命令如下：
```bash
git clone git@github.com:InternLM/InternLM.git --recurse-submodules
```

推荐使用 conda 构建一个 Python-3.10 的虚拟环境， 并基于`requirements/`文件安装项目所需的依赖包：
```bash
conda create --name internlm-env python=3.10 -y
conda activate internlm-env
cd internlm
pip install -r requirements/torch.txt 
pip install -r requirements/runtime.txt 
```

安装 flash-attention (version v1.0.5)：
```bash
cd ./third_party/flash-attention
python setup.py install
cd ./csrc
cd fused_dense_lib && pip install -v .
cd ../xentropy && pip install -v .
cd ../rotary && pip install -v .
cd ../layer_norm && pip install -v .
cd ../../../../
```

安装 Apex (version 23.05)：
```bash
cd ./third_party/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../../
```

### 环境镜像
用户可以使用提供的 dockerfile 结合 docker.Makefile 来构建自己的镜像，或者也可以从 https://hub.docker.com/r/internlm/internlm 获取安装了 InternLM 运行环境的镜像。

#### 镜像配置及构造
dockerfile 的配置以及构造均通过 docker.Makefile 文件实现，在 InternLM 根目录下执行如下命令即可 build 镜像：
``` bash
make -f docker.Makefile BASE_OS=centos7
``` 
在 docker.Makefile 中可自定义基础镜像，环境版本等内容，对应参数可直接通过命令行传递。对于 BASE_OS 分别支持 ubuntu20.04 和 centos7。

#### 镜像拉取
基于 ubuntu 和 centos 的标准镜像已经 build 完成也可直接拉取使用：

```bash
# ubuntu20.04
docker pull internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu20.04
# centos7
docker pull internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7
```

#### 容器启动
对于使用 dockerfile 构建或拉取的本地标准镜像，使用如下命令启动并进入容器：
```bash
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 20g --network=host --name myinternlm internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7 bash
```
容器内默认目录即 `/InternLM`，根据[使用文档](./usage.md)即可启动训练。
