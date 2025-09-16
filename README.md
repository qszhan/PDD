# PDD
Code (pytorch) for ['Progressive Dual-Space Discovering of Unknowns for Source-Free Open-Set Domain Adaptation'] on Digits (MNIST, USPS, SVHN), Office-31, Office-Home. This paper has been accepted by ECML-PKDD2025.  

## Preliminary

You need to download the [MNIST](https://github.com/myleott/mnist_png), [USPS](https://github.com/mingyuliutw/CoGAN), [SVHN](http://ufldl.stanford.edu/housenumbers/), [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) dataset, and modify the path of images in each `.txt` under the folder `./data/`.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.8.5
- pytorch == 1.13.1 
- torchvision == 0.14.1 
 
## Training and evaluation

Please refer to the file on [run.sh](run.sh).

---

## Citation

Zhan, Q., Wang, Q., Zeng, Xiao-Jun. Progressive Dual-Space Discovering of Unknowns for Source-Free Open-Set Domain Adaptation. *ECML-PKDD* (2025).  

---

## Acknowledgement

The code is based on  [SHOT (ICML 2020, also source-free)](https://arxiv.org/abs/2002.08546), and [TPDS](https://github.com/tntek/TPDS/blob/main/README.md).

---

## Contact

- [qianshan.zhan@manchester.ac.uk](mailto:qianshan.zhan@manchester.ac.uk)
