# better create a virtual environment with python 3.10 especially and activate it
py -3.10 -m venv venv
source venv/Scripts/activate
pip install wheel

# Install SAM
cd sam; pip install -e .
cd ..

# Install Grounding-Dino
git clone -b main https://github.com/IDEA-Research/GroundingDINO.git

############ IMPORTANT ###############
# install torch, torchvision and torchaudio manually if you have cuda or cpu, according to the cuda version
# from here https://pytorch.org/get-started/locally/

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
pip install scikit-image
pip install gradio==3.39.0
pip install gradio_client==0.5.0 wget gdown
pip install timm==0.4.5
pip install moviepy==1.0.3

# Install AST
git clone https://github.com/YuanGongND/ast.git ast_master
cp ./prepare.py ./ast_master

