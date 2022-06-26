#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn
import time
from models import ConvolutionalBlock,ResidualBlock
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import hickle as hkl
import numpy as np
import IPython.display as display
import os
import shutil

class Generator(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, n_blocks=5):
        super(Generator, self).__init__()
        self.conv_block1 = ConvolutionalBlock(in_channels=6, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=False, activation='relu')
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=kernel_size, n_channels=n_channels) for i in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=kernel_size,
                                              batch_norm=True, activation=None)
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=128, kernel_size=kernel_size,
                                              batch_norm=False, activation=None)
        self.conv_block4 = ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=kernel_size,
                                              batch_norm=False, activation=None)
        self.conv_block5 = ConvolutionalBlock(in_channels=256, out_channels=1, kernel_size=1,
                                              batch_norm=False, activation='tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)  # (batch_size, 1, 40, 40)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        sr_imgs = self.conv_block5(output)
        return sr_imgs

def make_input(imgs, distances): #imgs batchsize*1*40*40     distances batchsize*5
    dis = distances.unsqueeze(2).unsqueeze(3)
    dis = dis.repeat(1,1,40,40)
    data_input = torch.cat((imgs,dis),1)
    return data_input

class testDataset(Dataset):
    def __init__(self,path):
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
            label_one_hot = torch.zeros(5)
            label_one_hot[int(-abs((distance_chrome[i][0]))/40)]=1
            chrom = distance_chrome[i][1]
            self.sample_list.append([lr, hr, label_one_hot, dist, chrom])
#         print("dataset loaded : " + str(len(lo)) + '*' + str(len(lo[0])) + '*' + str(len(lo[0][0])) + '*' + str(len(lo[0][0][0])))

    def __getitem__(self, i):
        (lr_img, hr_img, label_one_hot, distance, chromosome) = self.sample_list[i]
        return lr_img, hr_img, label_one_hot, distance, chromosome

    def __len__(self):
        return len(self.sample_list)



# 模型参数

# device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(hklPath):
    shutil.rmtree('./static/data/picture')
    os.mkdir('./static/data/picture')
    checkpoint = torch.load("./model/G.pth")
    test_dataset = testDataset(hklPath)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True)
    generator = Generator(kernel_size=3,n_channels=64,n_blocks=5)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    cnt = 0
    start = time.time()
    SRs = []
    for i, (lr_img, hr_img, label, distance, __) in enumerate(test_loader):
        if i ==5 :
            break
        display.clear_output(wait=True)
        cnt = cnt + 1

        lr_img = lr_img.type(torch.FloatTensor).to(device)
        label = label.to(device)
        G_input = make_input(lr_img, label)
        with torch.no_grad():
            sr_img = generator(G_input.detach())
        sr=sr_img.squeeze().unsqueeze(2).to("cpu").numpy()
        SRs.append(sr)
        
        lr_img = sample_to_gray(lr_img)
        sr_img = sample_to_gray(sr_img)
        
        cmp = 'Reds'

        plt.imshow(lr_img, cmap=cmp , vmax=255,vmin=0)
        plt.savefig("./static/data/picture/" + str(cnt) + "_lr.png")
        plt.imshow(sr_img, cmap=cmp , vmax=255,vmin=0)
        plt.savefig("./static/data/picture/" + str(cnt) + "_sr.png")
    hkl.dump([SRs],'./static/data/high_resolution.hkl')
    return format(time.time()-start),len(test_dataset)

def sample_to_gray(img):
    img = img.squeeze().cpu()
    img = img.numpy()
    img = 127.5 * img + 127.5
    return img