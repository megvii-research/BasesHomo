# Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection [paper](https://arxiv.org/abs/2103.15346)
In this paper, we introduce a new framework for unsupervised deep homography estimation. Our contributions are 3 folds. First, unlike previous methods that regress 4 offsets for a homography, we propose a homography flow representation, which can be estimated by a weighted sum of 8 pre-defined homography flow bases. Second, considering a homography contains 8 Degree-of-Freedoms (DOFs) that is much less than the rank of the network features, we propose a Low Rank Representation (LRR) block that reduces the feature rank, so that features corresponding to the dominant motions are retained while others are rejected. Last, we propose a Feature Identity Loss (FIL) to enforce the learned image feature warp-equivariant, meaning that the result should be identical if the order of warp operation and feature extraction is swapped. With this constraint, the unsupervised optimization is achieved more effectively and more stable features are learned. Extensive experiments are conducted to demonstrate the effectiveness of all the newly proposed components, and results show that our approach outperforms the state-of-the-art on the homography benchmark datasets both qualitatively and quantitatively.

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


## Meta

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
