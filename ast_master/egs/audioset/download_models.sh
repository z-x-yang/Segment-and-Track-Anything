#!/bin/bash       
#title           :download_models.sh
#description     :
#author		     :Yuan Gong @ MIT
#time            :6/24/21 1:46 PM
#=======================================

# download all models to /ast/pretrained_models directory

# the best single model (with weight average)
wget https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1 -O ../../pretrained_models/audioset_10_10_0.4593.pth

# the model used for our ensemble exp
wget https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_10_10_0.4495.pth
wget https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_10_10_0.4483.pth
wget https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_10_10_0.4475.pth
wget https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_12_12_0.4467.pth
wget https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_14_14_0.4431.pth
wget https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1 -O ../../pretrained_models/ensemble/audioset_16_16_0.4422.pth