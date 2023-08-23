## Docker Images

### Image Configuration and Build
The configuration and build of the Dockerfile are implemented through the docker.Makefile. To build the image, execute the following command in the root directory of InternLM:
``` 
make -f docker.Makefile BASE_OS=centos7 DOCKERFILE_PATH=./dockerfile/Dockerfile-centos
``` 
In docker.Makefile, you can customize the basic image, environment version, etc., and the corresponding parameters can be passed directly through the command line. For BASE_OS, ubuntu20.04 and centos7 are respectively supported. The specific Dockerfile is specified through DOCKERFILE_PATH.

### Pull standard image
The standard image based on ubuntu and centos has been built and can be directly pulled:

```
# ubuntu20.04
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu20.04
# centos7
docker pull li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-centos7
```

### Run container
For the local standard image built with dockerfile or pulled, use the following command to run and enter the container:
```
docker run --gpus all -it -m 500g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --shm-size 10g --network=host --name myinternlm li126com/internlm:torch1.13.1-cuda11.7.1-flashatten1.0.5-ubuntu18.04 bash
```
The default directory in the container is `/InternLM`, please start training according to the README.
