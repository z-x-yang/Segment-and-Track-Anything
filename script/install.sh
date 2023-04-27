# Install SAM
cd sam; pip install -e .
cd -

# Install Grounding-Dino
cd src
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ../..

# Install other lib
pip install numpy opencv-python pycocotools matplotlib Pillow scikit-image
pip install gradio zip gdown ffmpeg

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -

