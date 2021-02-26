# 3D-HyperGAMO

# Generative Adversarial Minority Oversampling for Spectral-Spatial Hyperspectral Image Classification
The Code for "Generative Adversarial Minority Oversampling for Spectral-Spatial Hyperspectral Image Classification". [https://ieeexplore.ieee.org/document/9347550]
```
S. K. Roy, J. M. Haut, M. E. Paoletti, S. R. Dubey and A. Plaza. 
Generative Adversarial Minority Oversampling for Spectral-Spatial Hyperspectral Image Classification
IEEE Transactions on Geoscience and Remote Sensing
DOI: 10.1109/TGRS.2021.3052048
February 2021.
```

![3DGAMO](https://github.com/mhaut/3D-HyperGAMO/blob/master/images/gamo.png)


### Install and activate requires packages (with conda)
```
conda env create -f enviroment.yml
conda activate 3D-HyperGAMO
```

### Example of use
```
# Without datasets
git clone https://github.com/mhaut/hyperspectral_deeplearning_review/

# With datasets
git clone --recursive https://github.com/mhaut/hyperspectral_deeplearning_review/
cd HSI-datasets
python join_dsets.py
```


### Run code

```
python main.py --dataset IP 

```

Reference code: https://github.com/SankhaSubhra/GAMO
