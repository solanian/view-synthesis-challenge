# ARG VERSION=22.09
# FROM nvcr.io/nvidia/pytorch:${VERSION}-py3
FROM lukaszkucinski/cuda117_cudnn86

USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

## tzdata
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -qq -y \
    wget \
    git \
    build-essential \
	curl \
	bash-completion \
	htop \
	cmake \
    sudo \
	gawk \
	bison \
	git-lfs \
	tmux

RUN apt-get install -qq -y libgl1-mesa-glx libgeos-dev

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME

# Create the user
RUN groupadd $USERNAME \
    && useradd -g $USERNAME -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

CMD mkdir -p /workspace

# ARG CONDA_ENV_NAME="zipnerf"
# ARG CONDA_ENV_DIR="${CONDA_ENV_NAME}_env"
# RUN conda create -y -n ${CONDA_ENV_NAME} python=3.9

# # make the shell use the new conda envrionment by default temporarily
# SHELL ["conda", "run", "-n", "zipnerf", "/bin/bash", "-c"]

# Install ZIP-NeRF Requirements
WORKDIR /workspace
RUN git clone https://github.com/SuLvXiangXin/zipnerf-pytorch.git
WORKDIR /workspace/zipnerf-pytorch
RUN pip install -r requirements.txt
RUN pip install ./gridencoder

# Install nvdiffrast (optional, for textured mesh)
WORKDIR /workspace
RUN git clone https://github.com/NVlabs/nvdiffrast \
 && pip install ./nvdiffrast \
 && rm -rf nvdiffrast

# Install GeoNeRF Requirements
WORKDIR /workspace
RUN git clone https://github.com/idiap/GeoNeRF.git
WORKDIR /workspace/GeoNeRF
RUN pip install -r requirements.txt

# Install a specific cuda version of torch_scatter 
# see more detail at https://github.com/rusty1s/pytorch_scatter
# ARG CUDA_VERSION=11.8
# RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${CUDA_VERSION}.html

ARG CUDA=cu117
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html

RUN pip install gdown rembg[gpu,cli] torch==1.9.0 torchmetrics==0.6.0 test-tube

# Install tmux-beautify
RUN git clone https://github.com/gpakosz/.tmux.git ~/.oh-my-tmux \
	&& echo "set -g mouse on" >> ~/.oh-my-tmux/.tmux.conf \
	&& ln -s -f ~/.oh-my-tmux/.tmux.conf ~/.tmux.conf \
	&& cp ~/.oh-my-tmux/.tmux.conf.local ~/.tmux.conf.local

ADD .bashrc /home/$USERNAME
ENV SHELL /bin/bash