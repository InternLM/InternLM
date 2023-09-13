## Installation

### Environment Preparation
The required packages and corresponding version are shown as follows:
- Python == 3.10
- GCC == 10.2.0
- MPFR == 4.1.0
- CUDA >= 11.7
- Pytorch >= 1.13.1
- Transformers >= 4.28.0
- Flash-Attention >= v1.0.5
- Apex == 23.05
- GPU with Ampere or Hopper architecture (such as H100, A100)
- Linux OS

After installing the above dependencies, some system environment variables need to be updated:
```bash
export CUDA_PATH={path_of_cuda_11.7}
export GCC_HOME={path_of_gcc_10.2.0}
export MPFR_HOME={path_of_mpfr_4.1.0}
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${MPFR_HOME}/lib:${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
export PATH=${GCC_HOME}/bin:${CUDA_PATH}/bin:$PATH
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++
```

### Environment Installation
Clone the project `internlm` and its dependent submodules from the github repository, as follows:
```bash
git clone git@github.com:InternLM/InternLM.git --recurse-submodules
```

It is recommended to build a Python-3.10 virtual environment using conda and install the required dependencies based on the `requirements/` files:
```bash
conda create --name internlm-env python=3.10 -y
conda activate internlm-env
cd internlm
pip install -r requirements/torch.txt 
pip install -r requirements/runtime.txt 
```

Install flash-attention (version v1.0.5):
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

Install Apex (version 23.05):
```bash
cd ./third_party/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../../
```

### Environment Image
Users can use the provided dockerfile combined with docker.Makefile to build their own images, or obtain images with InternLM runtime environment installed from https://hub.docker.com/r/internlm/internlm.

#### Image Configuration and Build
The configuration and build of the Dockerfile are implemented through the docker.Makefile. To build the image, execute the following command in the root directory of InternLM:
``` bash
make -f docker.Makefile BASE_OS=centos7
``` 
In docker.Makefile, you can customize the basic image, environment version, etc., and the corresponding parameters can be passed directly through the command line. For BASE_OS, ubuntu20.04 and centos7 are respectively supported.

#### Pull Standard Image
The standard image based on ubuntu and centos has been built and can be directly pulled:

```bash
# ubuntu20.04
docker pull internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu20.04
# centos7
docker pull internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7
```

#### Run Container
For the local standard image built with dockerfile or pulled, use the following command to run and enter the container:
```bash
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 20g --network=host --name myinternlm internlm/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7 bash
```
The default directory in the container is `/InternLM`, please start training according to the [Usage](./usage.md).
