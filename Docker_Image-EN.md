## Standard Image Configuration and Usage
The configuration and usage of the Dockerfile are implemented through the docker.Makefile. To build the image, execute the following command:
``` 
make -f docker.Makefile
``` 
In docker.Makefile, you can customize the base image, environment version, etc. For BASE_OS, ubuntu and centos are supported respectively, and the corresponding standard dockerfile needs to be specified by -f in the build command.

## Supplementary environment installation
The installation and compilation of flashattention and apex need to rely on the GPU which is not supported during the docker build phase, so they need to be installed through the container. The process is as follows: 

(1) Start the container with the following command, where {image_name} should be replaced by the real image:
```
docker run --gpus all -it -m 200g  --cap-add=SYS_PTRACE  --cap-add=IPC_LOCK   --shm-size 4g  --network=host {image_name} bash
``` 

(2) Continue to install flashattention and apex in the container:
```
cd /InternLM/third_party/flash-attention && /opt/conda/bin/python setup.py install && cd ./csrc && cd fused_dense_lib && /opt/conda/bin/pip install -v . && cd ../xentropy && /opt/conda/bin/pip install -v . && cd ../rotary && /opt/conda/bin/pip install -v . && cd ../layer_norm && /opt/conda/bin/pip install -v . && cd ../../../../ && cd ./third_party/apex && /opt/conda/bin/pip --no-cache-dir install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && /opt/conda/bin/pip cache purge && rm -rf ~/.cache/pip
```

(3) Save the container as a new image:
```
docker commit {contrainer_id} {image_name}
```