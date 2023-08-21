## 标准镜像配置及使用
dockerfile的配置以及使用均通过docker.Makefile文件实现，执行如下命令即可build镜像：
``` 
make -f docker.Makefile
``` 
在docker.Makefile可自定义基础镜像，环境版本等内容，对于BASE_OS分别支持ubuntu和centos，在build命令中需要通过-f指定对应的标准dockerfile。

## 补充环境安装
flashattention和apex的安装编译需要依赖GPU，但是在docker build阶段目前不支持启用GPU，因此还需通过容器来安装，流程如下：  
（1）使用如下命令启动容器，其中{image_name}替换为真实的镜像：  
```
docker run --gpus all -it -m 200g  --cap-add=SYS_PTRACE  --cap-add=IPC_LOCK   --shm-size 4g  --network=host {image_name} bash
``` 
（2）在容器中继续安装flashattention和apex：
```
cd /InternLM/third_party/flash-attention && /opt/conda/bin/python setup.py install && cd ./csrc && cd fused_dense_lib && /opt/conda/bin/pip install -v . && cd ../xentropy && /opt/conda/bin/pip install -v . && cd ../rotary && /opt/conda/bin/pip install -v . && cd ../layer_norm && /opt/conda/bin/pip install -v . && cd ../../../../ && cd ./third_party/apex && /opt/conda/bin/pip --no-cache-dir install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && /opt/conda/bin/pip cache purge && rm -rf ~/.cache/pip
```
（3）将容器保存为新的镜像:
```
docker commit {contrainer_id} {image_name}
```