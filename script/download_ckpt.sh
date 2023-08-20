# Check and download aot-ckpt 
if [ ! -f ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth ]; then
    gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
else
    echo "R50_DeAOTL_PRE_YTB_DAV.pth already downloaded."
fi

# Check and download sam-ckpt
if [ ! -f ./ckpt/sam_vit_b_01ec64.pth ]; then
    wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "sam_vit_b_01ec64.pth already downloaded."
fi
