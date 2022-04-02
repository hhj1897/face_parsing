# RoI Tanh-polar Transformer Network for Face Parsing in the Wild

## Recent Updates

**2022-04-02 Update**: If you **could not download the weights with LFS**, check out issue https://github.com/hhj1897/face_parsing/issues/7#issuecomment-1086684106 for alternative downloading links.

**2022-03-04 Update**: We have released the FP-Age model which can perform face parsing and age estimation simultaneously, please visit https://github.com/ibug-group/fpage for details. 

---
Official repo for our paper *RoI Tanh-polar transformer network for face parsing in the wild*. 

__Note__: If you use this repository in your research, we kindly rquest you to cite the [following paper](https://arxiv.org/pdf/2102.02717):
```bibtex
@article{lin2021roi,
title = {RoI Tanh-polar transformer network for face parsing in the wild},
journal = {Image and Vision Computing},
volume = {112},
pages = {104190},
year = {2021},
issn = {0262-8856},
doi = {https://doi.org/10.1016/j.imavis.2021.104190},
url = {https://www.sciencedirect.com/science/article/pii/S0262885621000950},
author = {Yiming Lin and Jie Shen and Yujiang Wang and Maja Pantic},
keywords = {Face parsing, In-the-wild dataset, Head pose augmentation, Tanh-polar representation},
}
```

## Dependencies
* [git-lfs](https://git-lfs.github.com/)
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [ibug.roi_tanh_warping](https://github.com/ibug-group/roi_tanh_warping): See this repository for details: [https://github.com/ibug-group/roi_tanh_warping](https://github.com/ibug-group/roi_tanh_warping).
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).

## How to Install
```bash
git clone https://github.com/hhj1897/face_parsing
cd face_parsing
git lfs pull
pip install -e .
```

## How to Test
```bash
python face_warping_test.py -i 0 -e rtnet50 --decoder fcn -n 11 -d cuda:0
```
Command-line arguments:
```
-i VIDEO: Index of the webcam to use (start from 0) or
          path of the input video file
-d: Device to be used by PyTorch (default=cuda:0)
-e: Encoder (default=rtnet50)
--decoder: Decoder (default=fcn)
-n: Number of facial classes, can be 11 or 14 for now (default=11)
```
## iBugMask Dataset
The training and testing images, bounding boxes, landmarks, and parsing maps can be found in the following:

* [Google Drive](https://drive.google.com/file/d/1hGSki97qQPGNB812hh2Wf1_lP9NgJkti) 
* [ibug link](https://ibug.doc.ic.ac.uk/download/rtnet/ibugmask_release.zip)

## Label Maps

Label map for 11 classes: 
```
0 : background
1 : skin (including face and scalp)
2 : left_eyebrow
3 : right_eyebrow
4 : left_eye
5 : right_eye
6 : nose
7 : upper_lip
8 : inner_mouth
9 : lower_lip
10 : hair
```

Label map for 14 classes: 
```
0 : background
1 : skin (including face and scalp)
2 : left_eyebrow
3 : right_eyebrow
4 : left_eye
5 : right_eye
6 : nose
7 : upper_lip
8 : inner_mouth
9 : lower_lip
10 : hair
11 : left_ear
12 : right_ear
13 : glasses
```

## Visualisation
![](./imgs/vis1.jpg)
![](./imgs/vis2.jpg)
