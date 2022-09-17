# [ICCV 2021 Oral] Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Motion_Basis_Learning_for_Unsupervised_Deep_Homography_Estimation_With_Subspace_ICCV_2021_paper.pdf)

<h4 align="center">Nianjin Ye<sup>1</sup>, Chuan Wang<sup>1</sup>, Haoqiang Fan<sup>1</sup>, Shuaicheng Liu<sup>2,1</sup></center>
<h4 align="center">1. Megvii Research, 2. University of Electronic Science and Technology of China </center>

## Presentation Video
[[Youbube](https://www.youtube.com/watch?v=EcU02njjWHE)], [[Bilibili](https://www.bilibili.com/video/BV1ah411H7Th/)]

## Abstract
In this paper, we introduce a new framework for unsupervised deep homography estimation. Our contributions are 3 folds. First, unlike previous methods that regress 4 offsets for a homography, we propose a homography flow representation, which can be estimated by a weighted sum of 8 pre-defined homography flow bases. Second, considering a homography contains 8 Degree-of-Freedoms (DOFs) that is much less than the rank of the network features, we propose a Low Rank Representation (LRR) block that reduces the feature rank, so that features corresponding to the dominant motions are retained while others are rejected. Last, we propose a Feature Identity Loss (FIL) to enforce the learned image feature warp-equivariant, meaning that the result should be identical if the order of warp operation and feature extraction is swapped. With this constraint, the unsupervised optimization is achieved more effectively and more stable features are learned. Extensive experiments are conducted to demonstrate the effectiveness of all the newly proposed components, and results show that our approach outperforms the state-of-the-art on the homography benchmark datasets both qualitatively and quantitatively.

## Motivation
![motivation](https://user-images.githubusercontent.com/1344482/180999724-426f5396-8313-4893-9cbc-1943cf9b8b77.JPG)

## Basis Homo Representation
<img width="424" alt="basisHomo" src="https://user-images.githubusercontent.com/1344482/180954864-f9f0ac1b-f052-4a8f-bb3c-9f21f0ed46f3.png">

## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/181000099-9c1e10fa-2b33-42f2-9804-03e5fd196ac8.JPG)






## Requirements
- Python 3.5
- Pytorch 1.1.0
- Numpy 1.15.4

## Data pre-processing 

Please refer to [Content-Aware Unsupervised Deep Homography Estimation](https://github.com/JirongZhang/DeepHomography.git).

1. Download raw data
```sh
# GoogleDriver
https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing
# BaiduYun
https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA (key: gvor)
```
2. Unzip the data to directory "./dataset"

3. Run "video2img.py"

## Pretrained model

```sh
# GoogleDriver
https://1drv.ms/u/s!AglwI3TlqfaZhlU4BXo7iYiyLdI5?e=tCygKh
# BaiduYun
https://pan.baidu.com/s/1v4tF_RPEKgc6YF_mfRl7jQ (key: b3yq)
```

## Test
python evaluate.py

## Citation

All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Ye_2021_ICCV,
    author    = {Ye, Nianjin and Wang, Chuan and Fan, Haoqiang and Liu, Shuaicheng},
    title     = {Motion Basis Learning for Unsupervised Deep Homography Estimation With Subspace Projection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13117-13125}
}
```
