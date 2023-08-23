## Standard Image Configuration and Usage
The configuration and usage of the Dockerfile are implemented through the docker.Makefile. To build the image, execute the following command:
``` 
make -f docker.Makefile
``` 
In docker.Makefile, you can customize the base image, environment version, etc. For BASE_OS, ubuntu and centos are supported respectively, and the corresponding standard dockerfile needs to be specified by -f in the build command.

## Pull standard image
The standard image based on ubuntu and centos has been built and can be directly pulled:

ubuntu:
```
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu18.04
```
centos:
```
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7
```
## Run container
For the local standard image built with dockerfile or pulled, use the following command to run and enter the container:
```
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 10g --network=host --name myinternlm li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu18.04 bash
```
The default directory in the container is `/InternLM`, please start training according to the README.
