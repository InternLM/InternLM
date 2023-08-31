## Environment Image for torch2_flashatten2
This module is used to test the new version environment, the default test new environment is torch=2.0.1, flash-attention=2.1.0. 

### Build and Pull Image
When building the image, please make docker.Makefile in the InternLM root directory. This Makefile is shared with the standard environment image, and the Dockerfile used is located in the torch2_flashatten2 directory. You can also pull the image directly from https://hub.docker.com/r/internlm/internlm, the command is as follows:
```bash
# Build Image
# The network proxy uses system environment variables by default. If you need to manually set them, please pass in http_proxy and https_proxy in the command.
# ubuntu20.04
make -f docker.Makefile BASE_OS=ubuntu20.04 DOCKERFILE_PATH=./docker/torch2_flashatten2/Dockerfile-ubuntu CUDA_VERSION=11.8.0 PYTORCH_VERSION=2.0.1 TORCHVISION_VERSION=0.15.2 TORCHAUDIO_VERSION=2.0.2 FLASH_ATTEN_VERSION=2.1.0
# centos7
make -f docker.Makefile BASE_OS=centos7 DOCKERFILE_PATH=./docker/torch2_flashatten2/Dockerfile-centos CUDA_VERSION=11.8.0 PYTORCH_VERSION=2.0.1 TORCHVISION_VERSION=0.15.2 TORCHAUDIO_VERSION=2.0.2 FLASH_ATTEN_VERSION=2.1.0

# Pull Image
# ubuntu20.04
docker pull internlm/internlm:torch2.0.1-cuda11.8.0-flashatten2.1.0-ubuntu20.04 
# centos7
docker pull internlm/internlm:torch2.0.1-cuda11.8.0-flashatten2.1.0-centos7
```

### Run Container
For the local standard image built with dockerfile or pulled, use the following command to run and enter the container:
```bash
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 20g --network=host --name myinternlm internlm/internlm:torch2.0.1-cuda11.8.0-flashatten2.1.0-centos7 bash
```
The default directory in the container is `/InternLM`, please start training according to the [Usage](../doc/en/usage.md).