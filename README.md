# PDD
Code (pytorch) for ['Progressive Dual-Space Discovering of Unknowns for Source-Free Open-Set Domain Adaptation'] on Digits (MNIST, USPS, SVHN), Office-31, Office-Home. This paper has been accepted by ECML-PKDD2025.  

## Preliminary

You need to download the [MNIST](https://github.com/myleott/mnist_png), [USPS](https://github.com/mingyuliutw/CoGAN), [SVHN](http://ufldl.stanford.edu/housenumbers/), [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) dataset, and modify the path of images in each `.txt` under the folder `./data/`.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.3  
- pytorch == 1.6.0  
- torchvision == 0.7.0
 
