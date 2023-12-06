
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

No need to install zip due to its deprecation, instead, use this tip to handle Windows specific zip & rm problems:
https://github.com/z-x-yang/Segment-and-Track-Anything/issues/114

# Solve "zip" is not an internal or external command issue

* download zip and unzip from http://stahlworks.com/dev/index.php?tool=zipunzip
* put zip and unzip in a local folder
* add the folder path to system variable path

# Solve "rm" is not an internal or external command issue

* install MSYS2 from https://www.msys2.org/
* add msys64/usr/bin path to system variable path

# Other

* You can use bash after installing git, right-click in the folder and open git bash here (or in show more options)
* Run conda init bash, and then restart git bash to enter the virtual environment, but you need to use conda activate to activate (cannot use activate directly)
* For this project, you also need to pip install spatial_correlation_sampler. The error is not necessarily a torch version problem.
