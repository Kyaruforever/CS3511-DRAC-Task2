import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T 
from sklearn.model_selection import StratifiedKFold
import random
import torch
import torchvision.transforms.functional as F

class AddGaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, kfold=0, aug=True):
        self.trainaug = aug and train
        #path
        if train or val:
            imgPath = 'data/1. Original Images/a. Training Set/'
            gtPath = 'data/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
        else: #test
            imgPath = 'data/1. Original Images/b. Testing Set/'
        
        # prepare dataset
        self.imgs = [] #List[List[path,label,name]]
        if test:
            pathList = os.listdir(imgPath)
            pathList.sort(key=lambda x:int(x.split('.')[0]))
            for name in pathList:
                self.imgs.append([imgPath+name,-1,name])
        elif train or val:
            csvFile = pd.read_csv(gtPath)
            labels = []
            imgList = []
            for _, row in csvFile.iterrows():
                name = row['image name']
                label = int(row['image quality level'])
                labels.append(label)
                imgList.append([imgPath+name, label, name])
            skf = StratifiedKFold(n_splits=16) #no need for shuffle
            for index, (train_index, val_index) in enumerate(skf.split(np.zeros_like(labels),labels)):
                if index == kfold:
                    break
            if train:
                for i in train_index:
                    self.imgs.append(imgList[i])
            else:
                for i in val_index:
                    self.imgs.append(imgList[i])

        if self.trainaug:
            self.transform_weak,self.transform_strong=self.create_transforms()
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def create_transforms(self):
        data_aug_weak = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        }
        
        data_aug_strong = {
            'brightness': 0.5,  # how much to jitter brightness
            'contrast': 0.5,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        }

        transform_weak = T.Compose([
            T.Resize((420,420)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(
                size=((224,224)),
                scale=data_aug_strong['scale'],
                ratio=data_aug_strong['ratio']
            ),
            T.ColorJitter(
                brightness=data_aug_strong['brightness'],
                contrast=data_aug_strong['contrast'],
            ),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])
        
        transform_strong = T.Compose([
            T.Resize((420, 420)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(
                size=(224, 224),
                scale=data_aug_weak['scale'],
                ratio=data_aug_weak['ratio']
                ),
            T.ColorJitter(
                brightness=data_aug_weak['brightness'],
                contrast=data_aug_weak['contrast']
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(0.1, 0.05) # 添加高斯噪声
            ])
        
        return transform_weak, transform_strong
        

    def __getitem__(self, index):
        path, label, name = self.imgs[index]
        img = Image.open(path).convert('RGB')
        
        if not self.trainaug:
            img = self.transform(img)
        else:
            if label == 0:  # '0' for lower quality images
                img = self.transform_strong(img)
            else:
                img = self.transform_weak(img)

        return img, label, name
    
    
    def __len__(self):
        return len(self.imgs)


