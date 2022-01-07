import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import cv2
from config import mask_dir

CUT_PATCHES = 6

transform_masks = transforms.Compose([
    transforms.ToTensor()
])

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, masks=False):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))
        self.masks = masks
        if masks:
            self.mask_dir = mask_dir
            self.all_mask = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('L')
        tensor_image = self.transform(image)
        if self.masks:
            mask_name = self.all_imgs[idx].replace('.png', '_mask.png')
            if os.path.isfile(self.mask_dir + '/' + mask_name):
                mask_loc = os.path.join(self.mask_dir, mask_name)
                mask = Image.open(mask_loc).convert('L')
                tensor_mask = transform_masks(mask)
                return tensor_image, tensor_mask
        else:
            return tensor_image

    def getName(self, idx, mask=False):
        if mask:
            return self.all_imgs[idx].replace('.png', '_mask.png')
        else:
            return self.all_imgs[idx]


def augmentationDataset(dataset):
    ds = []
    widths = []
    for i in range(len(dataset)):
        j = dataset.__getitem__(i)
        ds.append(j)
        j = np.transpose(j.numpy(), (1, 2, 0))
        ds.append(torch.tensor(np.fliplr(j).copy()).permute(2, 0, 1))   # left-right flip
        ds.append(torch.tensor(np.flipud(j).copy()).permute(2, 0, 1))   # up-down flip
        blurred = gaussian_filter(j, sigma=0.5)         # blur
        ds.append(torch.tensor(blurred).permute(2, 0, 1))
        widths.append(j.shape[1])
        widths.append(j.shape[1])
        widths.append(j.shape[1])
        widths.append(j.shape[1])
    return ds, widths

def resize(dataset, original_width, original_height, dataset_masks=False):
    ds = []
    new_widths = []
    if not dataset_masks:
        for img in dataset:
            img_ = img.squeeze(0).numpy()
            img_ = cv2.normalize(img_, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img_ = img_.astype(np.uint8)
            
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_,(3,3), sigmaX=0, sigmaY=0) 

            # Canny Edge Detection
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

            # First method:
            # surface = []
            # for i in range(int(original_width / 3)):
            #     if edges[0, i] == 255:
            #         column = i
            #         surface.append(column)
            #         for j in range(1, original_height):
            #             condition = np.where(edges[j, column-1:column+2] == 255)
            #             if len(condition[0]) != 0:
            #                 column += condition[0][0] - 1
            #                 surface.append(column)
            #     if len(surface) != 0:
            #         break
            # cut = (int(max(surface)/16)+1) * 16

            # Second method:
            vector = np.zeros(original_width)
            for i in range(original_width):
                for j in range(original_height):
                    vector[i] += edges[j][i]
            derivative = np.gradient(vector)
            max = np.argmax(derivative)
            # min = np.argmin(derivative[max:]) + max
            cut = (int(max/16) + CUT_PATCHES) * 16

            crop_img = transforms.functional.crop(img, top=0, left=cut, height=original_height, width=(original_width-cut))
            new_widths.append(crop_img.shape[2])

            ds.append(crop_img)
    else:
        for img in dataset:
            img_ = img[0].squeeze(0).numpy()
            img_ = cv2.normalize(img_, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img_ = img_.astype(np.uint8)
            
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_, (3,3), sigmaX=0, sigmaY=0) 

            # Canny Edge Detection
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

            # Fist method:
            # surface = []
            # for i in range(int(original_width / 3)):
            #     if edges[0, i] == 255:
            #         column = i
            #         surface.append(column)
            #         for j in range(1, original_height):
            #             condition = np.where(edges[j, column-1:column+2] == 255)
            #             if len(condition[0]) != 0:
            #                 column += condition[0][0] - 1
            #                 surface.append(column)
            #     if len(surface) != 0:
            #         break
            # cut = (int(np.max(surface)/16)+3) * 16

            # Second method:
            vector = np.zeros(original_width)
            for i in range(original_width):
                for j in range(original_height):
                    vector[i] += edges[j][i]
            derivative = np.gradient(vector)
            max = np.argmax(derivative)
            # min = np.argmin(derivative[max:]) + max
            cut = (int(max/16) + CUT_PATCHES) * 16


            crop_img = transforms.functional.crop(img[0], top=0, left=cut, height=original_height, width=(original_width-cut))
            new_widths.append(crop_img.shape[2])
            # m = img[1]
            # m = m[:, :, cut:original_width]
            m = transforms.functional.crop(img[1], top=0, left=cut, height=original_height, width=(original_width-cut))
            ds.append([crop_img, m])
    return ds, new_widths