import os
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import cv2
import random       # shuffle
import shutil       # To copy the file
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
import backbone as b
from config import *


transform = transforms.Compose([
    transforms.ToTensor()
])


class NewDataSet(Dataset):
    def __init__(self, main_dir, masks=False):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))
        self.masks = masks
        if masks:
            self.mask_dir = newdataset_mask_dir
            self.all_mask = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        if self.masks:
            mask_name = self.all_imgs[idx].replace('.png', '_mask.png')
            if os.path.isfile(self.mask_dir + '/' + mask_name):
                mask_loc = os.path.join(self.mask_dir, mask_name)
                mask = Image.open(mask_loc).convert('L')
                tensor_mask = transform(mask)
                return tensor_image, tensor_mask
        else:
            return tensor_image

    def getName(self, idx, mask=False):
        if mask:
            return self.all_imgs[idx].replace('.png', '_mask.png')
        else:
            return self.all_imgs[idx]


def resizeNewDataset(dataset, dataset_masks=False):
    ds = []
    new_widths = []
    new_heights = []
    if not dataset_masks:
        for img in dataset:
            if img.shape[2] % patch_size != 0:              # width
                patches_in_image = int(np.floor(img.shape[2] / patch_size))
                new_width = img.shape[2] - (img.shape[2] - (patches_in_image * patch_size))
            else:
                new_width = img.shape[2]
            if img.shape[1] % patch_size != 0:              # height
                patches_in_image = int(np.floor(img.shape[1] / patch_size))
                new_height = img.shape[1] - (img.shape[1] - (patches_in_image * patch_size))
            else:
                new_height = img.shape[1]
            transform = transforms.CenterCrop((new_height, new_width))
            crop_img = transform(img)
            new_widths.append(new_width)
            new_heights.append(new_height)
            ds.append(crop_img)
    else:
        for img in dataset:
            if img[0].shape[2] % patch_size != 0:              # width
                patches_in_image = int(np.floor(img[0].shape[2] / patch_size))
                new_width = img[0].shape[2] - (img[0].shape[2] - (patches_in_image * patch_size))
            else:
                new_width = img[0].shape[2]
            if img[0].shape[1] % patch_size != 0:              # height
                patches_in_image = int(np.floor(img[0].shape[1] / patch_size))
                new_height = img[0].shape[1] - (img[0].shape[1] - (patches_in_image * patch_size))
            else:
                new_height = img[0].shape[1]
            transform = transforms.CenterCrop((new_height, new_width))
            crop_img = transform(img[0])
            m = transform(img[1])
            
            new_widths.append(new_width)
            new_heights.append(new_height)
            ds.append([crop_img, m])
    return ds, new_widths, new_heights




# --------------- Functions to create New Dataset ---------------

def BinarizeMasks(Mask_path):
    thresh = 128
    maxval = 255

    all_imgs = sorted(os.listdir(Mask_path))
    for i in all_imgs:
        im_gray = np.array(Image.open(Mask_path+i).convert('L'))
        im_bin = (im_gray > thresh) * maxval
        Image.fromarray(np.uint8(im_bin)).save(Mask_path+i)


def checkNewDataset():
    BinarizeMasks(newdataset_mask_dir)