# ACMMP
[News] The code for [ACMH](https://github.com/GhiXu/ACMH) is released!!!  
[News] The code for [ACMM](https://github.com/GhiXu/ACMM) is released!!!  
[News] The code for [ACMP](https://github.com/GhiXu/ACMP) is released!!!
## About
This repository contains the code for [Multi-Scale Geometric Consistency Guided and Planar Prior Assisted Multi-View Stereo](https://ieeexplore.ieee.org/document/9863705), which is the extension of ACMM and ACMP. If you find our work useful in your research, please consider citing:
```
@article{Xu2022Multi,
  title={Multi-Scale Geometric Consistency Guided and Planar Prior Assisted Multi-View Stereo},
  author={Xu, Qingshan and Kong, Weihang and Tao, Wenbing and Pollefeys, Marc},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
## Dependencies
The code has been tested on Ubuntu 16.04 with GTX 1080 Ti.  
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
