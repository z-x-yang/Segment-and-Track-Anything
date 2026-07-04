FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

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

COPY . .

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir \
        transformers==4.48.3 \
        hf_xet==1.5.1 \
        timm==1.0.27 \
        opencv-python==4.10.0.84 \
        Pillow==10.4.0 \
        gradio==3.39.0 \
        gradio_client==0.5.0 \
        pydantic==1.10.13 \
        fastapi==0.100.1 \
        starlette==0.27.0 \
        wget \
        gdown && \
    python -m pip install --no-cache-dir -e sam

CMD ["python", "app.py"]