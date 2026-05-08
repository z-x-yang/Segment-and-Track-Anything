# better create a virtual environment with python 3.10 especially and activate it
py -3.10 -m venv venv
source venv/Scripts/activate
pip install --upgrade pip setuptools wheel
# install torch and torchvision manually if you have cuda or cpu, according to the cuda version
# from here https://pytorch.org/get-started/locally/

# Install SAM
cd sam; pip install -e .
cd ..

# Install Grounding-Dino
git clone -b main https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
grep -vE "torch|torchvision" requirements.txt > require.txt
pip install -r require.txt
pip install -e . --no-build-isolation
cd ..

# Install other lib
pip install numpy opencv-python pycocotools matplotlib Pillow==9.2.0 scikit-image
pip install gradio==3.39.0 gdown ffmpeg==1.4
pip install timm==0.4.5
pip install wget
pip install moviepy==1.0.3

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
pip install .
cd ..

# Install AST
git clone https://github.com/YuanGongND/ast.git ast_master
cp ./prepare.py ./ast_master

