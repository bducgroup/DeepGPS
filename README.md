# DeepGPS
This repository is the official PyTorch implementation of **DeepGPS**, which has been published in _IEEE Transactions on Mobile Computing_ (TMC), a premier journal in the field of mobile computing.<br>

If you find **DeepGPS** useful, please cite our paper with the following bibtex:<br>

@article{liu2024deepgps,<br>
title={{DeepGPS}: deep learning enhanced {GPS} positioning in urban canyons},<br>
author={Liu, Zhidan and Liu, Jiancong and Xu, Xiaowen and Wu, Kaishun},<br>
journal={IEEE Transactions on Mobile Computing},<br>
volume={23},<br>
number={1},<br>
pages={376--392},<br>
year={2024},<br>
publisher={IEEE}<br>
}<br>

## Installation
Requirements: Python >= 3.5, Anaconda3
- Install Pytorch >= 1.8.1

` pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
`

The latest tested combination is: Python 3.8.5 + Pytorch 1.8.1 + cuda 11.1


## Quick Start
` 
python Predict.py
`

## Sample Description
The samples in stored as .npz file. which can be accessed by:
```
import numpy as np
npz_file = np.load('deepgps.npz')
```
Each .npz file contains contextual information and ground truth for one position point. The content in npz_file is following:
```
- npz_file['arr_0']：Environment Matrix.
- npz_file['arr_1']：Four float number. (corrected latitude,corrected longitude, original latitude, original longitude)
- npz_file['arr_2']：The number of seconds that have passed since 00:00:00, 20 June 2019.
- npz_file['arr_3']：The relative coordinate of corrected point (The original point is at (50,50)).
- npz_file['arr_4']：Gaussian Peak Representaiton for ground truth.
- npz_file['arr_5']：Skyplot Matrix.
```

## Download
The weights of a well-trained model can be downloaded from Baidu Cloud Storage (百度网盘) using the following information:<br>

链接：https://pan.baidu.com/s/1ehZHT895ass9PEZ3AsFaTQ?pwd=jb0w<br>
提取码：jb0w<br>

## Other Works

Please follow our group at https://github.com/SZU-BDUC/.
