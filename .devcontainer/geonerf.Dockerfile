ARG VERSION=21.09
FROM nvcr.io/nvidia/pytorch:${VERSION}-py3
# FROM pengjl929/cu11-torch1.9:latest

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

# Install GeoNeRF Requirements
WORKDIR /workspace
RUN git clone https://github.com/idiap/GeoNeRF.git
WORKDIR /workspace/GeoNeRF
RUN pip install -r requirements.txt

RUN pip install gdown test-tube torchmetrics==0.6.0 scikit-image==0.16.2 wandb typing-extensions==4.6.3 plyfile plotly kaleido

# Install nvtop
RUN git clone -b 2.0.4 https://github.com/Syllo/nvtop.git && \
    mkdir -p nvtop/build && cd nvtop/build && \
    cmake .. && \
    make && \
    sudo make install && \
    cd ../.. && \
    rm -rf nvtop

# Install tmux-beautify
RUN git clone https://github.com/gpakosz/.tmux.git ~/.oh-my-tmux \
	&& echo "set -g mouse on" >> ~/.oh-my-tmux/.tmux.conf \
	&& ln -s -f ~/.oh-my-tmux/.tmux.conf ~/.tmux.conf \
	&& cp ~/.oh-my-tmux/.tmux.conf.local ~/.tmux.conf.local

ADD .bashrc /home/$USERNAME
ENV SHELL /bin/bash