# Install SAM
cd sam; pip install -e .
cd -

# Install Grounding-Dino
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO

# Install other lib
pip install numpy opencv-python pycocotools matplotlib Pillow scikit-image
pip install gradio zip gdown ffmpeg
pip install gradio ##### for some reason, in the jupyter notebook needs to be installed again
pip install imageio==2.26.0 #the current code does not work with imageio, need to install old version

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -

