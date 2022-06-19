import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import math

def label_parse(label):
    f = open(label, 'r', encoding = 'utf-8')
    for line in f:
        pass    
    f.close()
    line = line.split(', ')
    return np.array([float(i) for i in line])
    
    #return np.array([float(line[0]),float(line[1]),float(line[4]),float(line[5])])
    #return np.array([float(line[1]), float(line[5])])
    #return np.array([float(line[0]),float(line[1])])
    #return np.array([float(line[0])])

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['img'], sample['label']
        imgs = image / 255.0
        imgs = np.transpose(imgs, (2,0,1))
        sample['img'] = torch.from_numpy(imgs)
        sample['label'] = torch.from_numpy(label)
        return {'img': sample['img'], 'label': sample['label']}

class ScrewholeDataset(Dataset):
    def __init__(self, image_dir, transform=transforms.Compose([ToTensor()])):
        """
        Args:
            image_dir: image와 label이 같은 폴더에 있음
            transform: 전처리
        """
        self.img_dir = image_dir
        self.transform = transform
        self.img_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.label_list = glob.glob(os.path.join(self.img_dir, '*.txt'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        image_ = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image_, (512, 512))
        label_name = self.label_list[idx]
        label = label_parse(label_name)
        label_1 = np.array([label[0], label[1], label[4], label[5]]) # stage 1 takes "x", "y"
        normalize_ratio = 1.0/100.0
        label_2 = np.array([label[2], label[6]]) # stage 2 takes "a"
        #label_2 = np.array([math.atan(label[2]), math.atan(label[6])]) # stage 2 takes "a"
        print("image_name: ", img_name)
        for i in range(len(label_1)):
            label_1[i] = label_1[i] * normalize_ratio
        sample = {'img': image, 'label': label_1}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, image, label_2