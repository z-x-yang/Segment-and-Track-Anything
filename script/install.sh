# better create a virtual environment with python 3.10 especially and activate it
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools==80.9.0

# Install SAM
cd sam; pip install -e .
cd ..

# Install Grounding-Dino. IF you get that "Failed to load custom C++" refer to
# https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708

git clone -b main https://github.com/IDEA-Research/GroundingDINO.git

############ IMPORTANT ###############

# If your CUDA version is different, use the matching command from:
# https://pytorch.org/get-started/locally/

# The versions below (2.0.1 / 0.15.2 / 2.0.2) are recommended and should be kept the same.
# Only change the wheel index (cu118, cu121, cpu, etc.).

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2
pip install addict==2.4.0
pip install yapf==0.40.2
pip install timm==0.4.5
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install Pillow==10.4.0
pip install scikit-image==0.24.0
pip install matplotlib==3.9.2
pip install supervision==0.22.0
pip install pycocotools==2.0.8

cd GroundingDINO
pip install -e . --no-build-isolation
cd ..

# Install other lib
pip install \
gradio==3.39.0 \
gradio_client==0.5.0 \
pydantic==1.10.13 \
fastapi==0.100.1 \
starlette==0.27.0 \
wget \
gdown
pip install moviepy==1.0.3

# Install AST
git clone https://github.com/YuanGongND/ast.git ast_master
cp ./prepare.py ./ast_master

