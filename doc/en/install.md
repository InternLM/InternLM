## InternLM Installation

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
Users can obtain an image with the InternLM runtime environment installed from https://hub.docker.com/r/sunpengsdu/internlm. The commands for pulling the image and starting the container are as follows:

```bash
# pull image
docker pull sunpengsdu/internlm:torch2.0.1-cuda11.8-flashatten2.0.0-centos
# start container
docker run --gpus all -d -it --shm-size=2gb sunpengsdu/internlm:torch2.0.1-cuda11.8-flashatten2.0.0-centos
docker exec -it mytorch2.0 bash
```