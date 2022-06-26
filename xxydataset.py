# -*- encoding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import hickle as hkl

# 基于Dataset类，自定义的数据集加载器，
path = './data/GM12878/train_data_half.hkl'
class xxyDataset(Dataset):
    def __init__(self):
        lo, hi, distance_chrome = hkl.load(path)
        lo = lo.squeeze()
        hi = hi.squeeze()
        lo = np.expand_dims(lo,axis=1)
        hi = np.expand_dims(hi,axis=1)
        
        self.sample_list = []
        for i in range(len(lo)):
            lr = lo[i]
            hr = hi[i]
            dist = distance_chrome[i][0]
            chrom = distance_chrome[i][1]
            self.sample_list.append([lr, hr, dist, chrom])
        print("dataset loaded : " + str(len(lo)) + '*' + str(len(lo[0])) + '*' + str(len(lo[0][0])) + '*' + str(len(lo[0][0][0])))

    def __getitem__(self, i):
        (lr_img, hr_img, distance, chromosome) = self.sample_list[i]
        return lr_img, hr_img, distance, chromosome

    def __len__(self):
        return len(self.sample_list)
