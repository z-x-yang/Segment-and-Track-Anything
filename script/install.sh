# better create a virtual environment with python 3.12 especially and activate it
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Install SAM
cd sam; python -m pip install -e .
cd ..

############ IMPORTANT ###############

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/

# The Torch versions below are recommended and should be kept the same.
# Only change the wheel index (cu118, cu121, cu124, cu128, cpu, etc.).

python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers==4.48.3
python -m pip install hf_xet==1.5.1
python -m pip install timm==1.0.27
python -m pip install opencv-python==4.10.0.84
python -m pip install Pillow==10.4.0

# Install other lib
python -m pip install \
gradio==3.39.0 \
gradio_client==0.5.0 \
pydantic==1.10.13 \
fastapi==0.100.1 \
starlette==0.27.0 \
wget \
gdown

# Install AST
git clone https://github.com/YuanGongND/ast.git ast_master
cp ./prepare.py ./ast_master