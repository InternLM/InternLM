DOCKER_REGISTRY          ?= docker.io
DOCKER_ORG               ?= my
DOCKER_IMAGE             ?= internlm
DOCKER_FULL_NAME          = $(DOCKER_REGISTRY)/$(DOCKER_ORG)/$(DOCKER_IMAGE)

CUDA_VERSION              = 11.7.1
GCC_VERSION               = 10.2.0

CUDNN_VERSION             = 8
BASE_RUNTIME              =
# ubuntu20.04  centos7
BASE_OS                   = centos7
BASE_DEVEL                = nvidia/cuda:$(CUDA_VERSION)-cudnn$(CUDNN_VERSION)-devel-${BASE_OS}
# The conda channel to use to install cudatoolkit
CUDA_CHANNEL              = nvidia
# The conda channel to use to install pytorch / torchvision
INSTALL_CHANNEL          ?= pytorch

PYTHON_VERSION           ?= 3.10
PYTORCH_VERSION          ?= 1.13.1
TORCHVISION_VERSION      ?= 0.14.1
TORCHAUDIO_VERSION       ?= 0.13.1
BUILD_PROGRESS           ?= auto
TRITON_VERSION           ?=
GMP_VERSION              ?= 6.2.1
MPFR_VERSION             ?= 4.1.0
MPC_VERSION              ?= 1.2.1
GCC_VERSION              ?= 10.2.0
HTTPS_PROXY_I            ?=
HTTP_PROXY_I             ?=
FLASH_ATTEN_VERSION      ?= 1.0.5
FLASH_ATTEN_TAG          ?= v${FLASH_ATTEN_VERSION}

BUILD_ARGS                = --build-arg BASE_IMAGE=$(BASE_IMAGE) \
                            --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
                            --build-arg CUDA_VERSION=$(CUDA_VERSION) \
                            --build-arg CUDA_CHANNEL=$(CUDA_CHANNEL) \
                            --build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
                            --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
                            --build-arg TORCHAUDIO_VERSION=$(TORCHAUDIO_VERSION) \
                            --build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
                            --build-arg TRITON_VERSION=$(TRITON_VERSION) \
                            --build-arg GMP_VERSION=$(GMP_VERSION) \
                            --build-arg MPFR_VERSION=$(MPFR_VERSION) \
                            --build-arg MPC_VERSION=$(MPC_VERSION) \
                            --build-arg GCC_VERSION=$(GCC_VERSION) \
                            --build-arg https_proxy=$(HTTPS_PROXY_I) \
                            --build-arg http_proxy=$(HTTP_PROXY_I) \
                            --build-arg FLASH_ATTEN_TAG=$(FLASH_ATTEN_TAG)

EXTRA_DOCKER_BUILD_FLAGS ?=

BUILD                    ?= build
# Intentionally left blank
PLATFORMS_FLAG           ?=
PUSH_FLAG                ?=
USE_BUILDX               ?=1
BUILD_PLATFORMS          ?=
WITH_PUSH                ?= false
BUILD_TYPE               ?= intrenlm-dev

# Setup buildx flags
ifneq ("$(USE_BUILDX)","")
BUILD                     =  buildx build
ifneq ("$(BUILD_PLATFORMS)","")
PLATFORMS_FLAG            = --platform="$(BUILD_PLATFORMS)"
endif
endif
# endif

# # Only set platforms flags if using buildx
# ifeq ("$(WITH_PUSH)","true")
# PUSH_FLAG               = --push
# endif
# endif

ifeq ($(findstring centos,$(BASE_OS)),centos)
    DOCKERFILE_PATH ?= ./docker/Dockerfile-centos
else
    DOCKERFILE_PATH ?= ./docker/Dockerfile-ubuntu
endif

#use -f to specify dockerfile
DOCKER_BUILD              = DOCKER_BUILDKIT=1 \
                            docker $(BUILD) \
                                   --progress=$(BUILD_PROGRESS) \
                                   $(EXTRA_DOCKER_BUILD_FLAGS) \
                                   $(PLATFORMS_FLAG) \
                                   $(PUSH_FLAG) \
                                   -f $(DOCKERFILE_PATH) \
                                   -t $(DOCKER_FULL_NAME):$(DOCKER_TAG) \
                                   $(BUILD_ARGS) .

                                   # --target $(BUILD_TYPE)

.PHONY: all
all: devel-image

.PHONY: devel-image
devel-image: BASE_IMAGE := $(BASE_DEVEL)
devel-image: DOCKER_TAG := torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}-flashatten${FLASH_ATTEN_VERSION}-${BASE_OS}
devel-image:
	$(DOCKER_BUILD)

.PHONY: clean
clean:
	-docker rmi -f $(shell docker images -q $(DOCKER_FULL_NAME))