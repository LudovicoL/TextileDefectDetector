#!/bin/bash
if [ ! -d "../dataset" ]; then
  mkdir "../dataset"
fi
cd ../dataset
mkdir AITEX
cd ./AITEX
wget https://www.aitex.es/wp-content/uploads/2019/07/Defect_images.7z
wget https://www.aitex.es/wp-content/uploads/2019/07/NODefect_images.7z
wget https://www.aitex.es/wp-content/uploads/2019/07/Mask_images.7z
if ! command -v 7z &> /dev/null
then
    echo "7z could not be found"
    sudo apt-get install p7zip-full
fi
7z x Defect_images.7z
7z x NODefect_images.7z
7z x Mask_images.7z
rm Defect_images.7z
rm NODefect_images.7z
rm Mask_images.7z
#rm ./Defect_images/0094_027_05.png
#rm ./Mask_images/0094_027_05_mask.png
echo Archives deleted successfully.
cd ../../utils
python CreateDataset.py