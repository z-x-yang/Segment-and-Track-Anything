FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    wget \
    curl \
    ca-certificates \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip wheel setuptools

RUN pip install \
    transformers==4.30.2 \
    addict==2.4.0 \
    yapf==0.40.2 \
    timm==0.4.5 \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    Pillow==10.4.0 \
    scikit-image==0.24.0 \
    matplotlib==3.9.2 \
    supervision==0.22.0 \
    pycocotools==2.0.8 \
    gradio==3.39.0 \
    gradio_client==0.5.0 \
    pydantic==1.10.13 \
    fastapi==0.100.1 \
    starlette==0.27.0 \
    wget \
    gdown

RUN pip install -e sam

RUN git clone -b main https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    pip install -e . --no-build-isolation

WORKDIR /workspace

CMD ["python", "app.py"]