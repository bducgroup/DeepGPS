# DeepGPS
This repository is the official PyTorch implementation of DeepGPS:

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