# MoSID-Breast-Tumor-Segmentation-from-Multi-parametric-MRI

## Paper:
Please see:
* Confernce paper: MoSID: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation (https://link.springer.com/chapter/10.1007/978-3-031-45350-2_8)
* Journal paper: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation and Computer-aided Diagnosis (https://ieeexplore.ieee.org/document/10388458)

## Introduction:
This project includes both train/test code for training the MoSID framwork.

![image](https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation/blob/main/Img/Results.png)

## Requirements:
* python 3.10
* pytorch 1.12.1
* numpy 1.23.3
* tensorboard 2.10.1
* simpleitk 2.1.1.1
* scipy 1.9.1

## Setup

### Dataset
* For training the segmentation models, you need to put the data in this format：

```
./data
├─train.txt
├─test.txt
├─Guangdong
      ├─Guangdong_1
          ├─P0.nii.gz
          ├─P1.nii.gz
          ├─P2.nii.gz
          ├─P3.nii.gz
          ├─P4.nii.gz     
          └─P5.nii.gz
      ├─Guangdong_2
      ├─Guangdong_3
      ...
├─Guangdong_breast
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...
├─Guangdong_gt
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...         
└─Yunzhong
└─Yunzhong_breast
└─Yunzhong_gt
└─Ruijin
└─Ruijin_breast
└─Ruijin_gt
...
```
* The format of the train.txt / test.txt is as follow：
```
./data/train.txt
├─'Guangdong_1'
├─'Guangdong_2'
├─'Guangdong_3'
...
├─'Yunzhong_100'
├─'Yunzhong_101'
...
├─'Ruijin_1010'
...
```

## Citation
If you find the code or data useful, please consider citing the following papers:

* Zhang et al., MoSID: Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation, MICCAI Workshop on Cancer Prevention through Early Detection (2023), https://doi.org/10.1007/978-3-031-45350-2_8
* Chen et al., Modality-Specific Information Disentanglement from Multi-parametric MRI for Breast Tumor Segmentation and Computer-aided Diagnosis, IEEE Transactions on Medical Imaging (2023), https://doi.org/10.1109/TMI.2024.3352648
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001






