
# Install SAM & its dependencies

cd sam; pip install -e .
cd ..
pip install opencv-python pycocotools matplotlib onnxruntime onnx

# Install pytorch according to PyTorch official guidance

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

# Install Pytorch Correlation from source

git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -

# Install Grounding-Dino

pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO

# Install other lib

pip install numpy Pillow scikit-image
pip install gradio==3.38.0 gdown ffmpeg  

No need to install zip due to its deprecation: 