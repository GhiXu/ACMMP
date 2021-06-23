# ACMMP
[News] The code for [ACMH](https://github.com/GhiXu/ACMH) is released!!!  
[News] The code for [ACMM](https://github.com/GhiXu/ACMM) is released!!!  
[News] The code for [ACMP](https://github.com/GhiXu/ACMP) is released!!!
## About
This repository contains the code for Multi-Scale Geometric Consistency Guided and Planar Prior Assisted Multi-View Stereo, which is the extension of ACMM and ACMP.
## Dependencies
The code has been tested on Ubuntu 14.04 with GTX 1080 Ti.  
* [Cuda](https://developer.nvidia.com/zh-cn/cuda-downloads) >= 6.0
* [OpenCV](https://opencv.org/) >= 2.4
* [cmake](https://cmake.org/)
## Usage
* Complie ACMMP
```  
cmake .  
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to ACMMP input   
Run ./ACMMP $data_folder to get reconstruction results
```
## Acknowledgemets
This code largely benefits from the following repositories: [Gipuma](https://github.com/kysucix/gipuma) and [COLMAP](https://colmap.github.io/). Thanks to their authors for opening source of their excellent works.
