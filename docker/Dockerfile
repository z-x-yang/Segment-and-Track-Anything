FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Use the full arch list may take more time to build
# ARG TORCH_CUDA_ARCH_LIST="Turing"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# Set environment variables.
## CUDA architectures
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
## User ID
ENV USER_ID=1000

# Install required apt packages and clear cache afterwards.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    nano \
    sudo \
    vim \
    python3-opencv \
    ca-certificates \
    python3-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER ${USER_ID}
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

ADD . /home/user/cc
USER root
RUN chown -R user /home/user/cc
USER ${USER_ID}

# Upgrade pip and install packages.
RUN python3 -m pip install --upgrade pip setuptools pathtools promise pybind11 numpy opencv-python pycocotools matplotlib scikit-image Pillow ffmpeg gdown gradio
RUN pip install -e /home/user/cc/sam
RUN pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO
RUN pip install spatial-correlation-sampler
# Change working directory
WORKDIR /home/user/cc

