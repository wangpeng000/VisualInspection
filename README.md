# SCL-VI: Self-supervised Context Learning for Visual Inspection of Industrial Defects
<a href="https://arxiv.org/abs/2311.06504"><img src="https://img.shields.io/badge/arXiv-2311.06504-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/github/license/WU-CVGL/BAD-NeRF" height=22.5></a>

We address the challenge of detecting object defects through the self-supervised learning approach of solving the jigsaw puzzle problem.

## Results
![segmentation](./doc/svdd_result.jpeg)

## Dependencies
Since I did this project a long time ago, there may be some potential issues with environmental dependencies.
- Tested with Python 3.8
- [Pytorch](http://pytorch.org/) v1.6.0

## Dateset
- Dataset : [MvTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

## Run Training
- python train.py --obj=cable --lambda_value=1 --D=64 --epoches=400 --lr=1e-4 --gpu=0

## Run Affinity Testing
- python test.py --obj=cable --gpu=0
- enc.load(obj, N) N is the serial number of the obtained training weight file

## Anomaly maps
- python heat_map.py --obj=cable
- enc.load(obj, N) N is the serial number of the obtained training weight file

## Details:
- The input of the network should be 256x256
- data.npy contains the relative positions and their reference numbers.