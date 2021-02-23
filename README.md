# ibug.face_parsing
RoI Tanh-polar Transformer Network for Face Parsing in the Wild.


__Note__: If you use this repository in your research, we kindly rquest you to cite the [following paper](https://arxiv.org/pdf/2102.02717):
```bibtex
@misc{lin2021roi,
      title={RoI Tanh-polar Transformer Network for Face Parsing in the Wild}, 
      author={Yiming Lin and Jie Shen and Yujiang Wang and Maja Pantic},
      year={2021},
      eprint={2102.02717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dependencies
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
python face_warping_test.py -i 0
```

Command-line arguments:
```
-i VIDEO: Index of the webcam to use (start from 0) or
          path of the input video file
-d: Device to be used by PyTorch (default=cuda:0)
-b: Enable benchmark mode for CUDNN
```

## Visualisation
![](./imgs/vis1.jpg)
![](./imgs/vis2.jpg)