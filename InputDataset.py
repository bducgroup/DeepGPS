import math
from math import cos, sin, pi
import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader,random_split

class PreDataset(Dataset):
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
        self.pos = 0
        self.length = len(self.samples_paths)
        self.mistake = 0

    def __len__(self):
        return len(self.samples_paths)

    def load(self, index):
        """
            feature1: environment matrix
            feature2: timestamp matrix
            feature3: skyplot matrix
            label1: Gaussian Peak representation of Ground Truth
            label2: Real one-hot vector for position error distance
        """
        try:
            filename = self.samples_paths[index]
            path = self.samples_dir + self.samples_paths[index]
            arr = np.load(path)
            input1 = [None]
            input3 = [None]
            input1[0] = np.array(arr['arr_0'], dtype='float32') / 351
            input3[0] = np.array(arr['arr_5'], dtype='float32')
            gps_time = arr['arr_2'][0]
            time_by_day1 = (gps_time % 86400) / 86400  # Period is 86400
            time_by_day2 = (gps_time % 86154) / 86154  # Period is（24*3600-246）
            if time_by_day1 < 0.5:
                time_by_noon = 0
            else:
                time_by_noon = 1
            time_x1 = cos(2 * pi * time_by_day1)
            time_y1 = sin(2 * pi * time_by_day1)
            time_x2 = cos(2 * pi * time_by_day2)
            time_y2 = sin(2 * pi * time_by_day2)
            input2 = np.asarray([time_by_day1, time_by_day2,time_x1, time_y1, time_x2, time_y2,  time_by_noon],
                                dtype='float32')
            feature1 = torch.Tensor(input1)
            feature2 = torch.Tensor(input2)
            feature3 = torch.Tensor(input3)
            label1 = torch.Tensor(arr['arr_4'])
            distance = np.sqrt(((arr['arr_3'] - 50) ** 2).sum(0))
            if distance>=50: distance = 48
            distance = math.floor(distance/2.0)
            label2 = torch.LongTensor([distance])
            del input1
            del input2
            del input3
            del gps_time
            del arr
            return ((feature1, feature2, feature3), (label1, label2,filename))
        except:
            self.mistake += 1
            print('-----------------------------------------------------mistake:', path)

            return None

    def __getitem__(self, index):
        return self.load(index)


