# TextileDefectDetector
Textile Defect Detection with Variational Autoencoder and Gaussian Mixture Model.
PLEASE NOTE: The implementation is not complete.

## Dataset
**AITEX**: Download from [AITEX website](https://www.aitex.es/afid/). For automatic download:
```
cd utils/
./get_aitex.sh
cd ..
```

## How to use
To train the model, use:
```
python train_network.py
```
To evaluate the model:
```
python test_network.py
```


## Minimum requirements
Python 3.9 with PyTorch 1.9.0. Use the file ```environment.yml``` for the conda environment.