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

import backbone as b
from config import *


transform = transforms.Compose([
    transforms.ToTensor()
])


class AitexDataSet(Dataset):
    def __init__(self, main_dir, masks=False):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))
        self.masks = masks
        if masks:
            self.mask_dir = aitex_mask_dir
            self.all_mask = sorted(os.listdir(self.mask_dir))

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
                tensor_mask = transform(mask)
            else:
                tensor_mask = torch.zeros([1, 256, 4096])
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
    heights = []
    for i in range(len(dataset)):
        j = dataset.__getitem__(i)

        ds.append(j)                                                    # Original image
        widths.append(j.shape[2])
        heights.append(j.shape[1])

        noised_image = b.add_noise(j, noise_factor=0.05)                # Gaussian noise
        ds.append(noised_image)
        widths.append(noised_image.shape[2])
        heights.append(noised_image.shape[1])

        j = np.transpose(j.numpy(), (1, 2, 0))
        ds.append(torch.tensor(np.fliplr(j).copy()).permute(2, 0, 1))   # orizontal flip
        widths.append(j.shape[1])
        heights.append(j.shape[0])

        ds.append(torch.tensor(np.flipud(j).copy()).permute(2, 0, 1))   # vertical flip
        widths.append(j.shape[1])
        heights.append(j.shape[0])

        blurred = gaussian_filter(j, sigma=0.5)                         # blur
        ds.append(torch.tensor(blurred).permute(2, 0, 1))
        widths.append(j.shape[1])
        heights.append(j.shape[0])
        
    return ds, widths, heights

def resizeAitex(dataset, original_width, original_height, dataset_masks=False):
    ds = []
    new_widths = []
    new_heights = []
    if not dataset_masks:
        for img in dataset:
            img_ = img.squeeze(0).numpy()
            img_ = cv2.normalize(img_, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img_ = img_.astype(np.uint8)
            
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_,(3,3), sigmaX=0, sigmaY=0) 

            # Canny Edge Detection
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

            vector = np.zeros(original_width)
            for i in range(original_width):
                for j in range(original_height):
                    vector[i] += edges[j][i]
            derivative = np.gradient(vector)
            max = np.argmax(derivative)
            cut = (int(max/patch_size) + CUT_PATCHES) * patch_size

            crop_img = transforms.functional.crop(img, top=0, left=cut, height=original_height, width=(original_width-cut))
            new_widths.append(crop_img.shape[2])
            new_heights.append(crop_img.shape[1])

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

            vector = np.zeros(original_width)
            for i in range(original_width):
                for j in range(original_height):
                    vector[i] += edges[j][i]
            derivative = np.gradient(vector)
            max = np.argmax(derivative)
            cut = (int(max/patch_size) + CUT_PATCHES) * patch_size

            crop_img = transforms.functional.crop(img[0], top=0, left=cut, height=original_height, width=(original_width-cut))
            new_widths.append(crop_img.shape[2])
            new_heights.append(crop_img.shape[1])

            m = transforms.functional.crop(img[1], top=0, left=cut, height=original_height, width=(original_width-cut))
            ds.append([crop_img, m])
    return ds, new_widths, new_heights




# --------------- Functions to create Aitex Dataset ---------------

def Reformat_Image(ImageFilePath, new_width, new_height, color, offset):

    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if color == 'white':
        color = (255, 255, 255, 255)
    elif color == 'black':
        color = (0, 0, 0, 255)

    if offset == 'center':
        offset = (int(round(((new_width - width) / 2), 0)), int(round(((new_height - height) / 2), 0)))
    elif offset == 'right':
        offset = (0, 0)
    elif offset == 'left':
        offset = ((new_width - width), (new_height - height))

    background = Image.new('RGBA', (new_width, new_height), color)

    background.paste(image, offset)
    background.save(ImageFilePath)
    # print("Image " + ImageFilePath + " has been resized!")

def DeleteFolder(path):
    shutil.rmtree(path)
    # print(path + ' deleted!')

def MergeMasks(name):
    mask1 = Image.open(name+'_mask1.png').convert('L')
    mask2 = Image.open(name+'_mask2.png').convert('L')
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask = np.add(mask1, mask2)
    mask = Image.fromarray(mask)
    mask.save(name+'_mask.png',"PNG")
    os.remove(name+'_mask1.png')
    os.remove(name+'_mask2.png')

def BinarizeMasks(Mask_path):
    thresh = 128
    maxval = 255

    all_imgs = sorted(os.listdir(Mask_path))
    for i in all_imgs:
        im_gray = np.array(Image.open(Mask_path+i).convert('L'))
        im_bin = (im_gray > thresh) * maxval
        Image.fromarray(np.uint8(im_bin)).save(Mask_path+i)

def FlipImage(filename):
    image = Image.open(filename)
    image = np.fliplr(image)
    Image.fromarray(np.uint8(image)).save(filename)

def CreateAitexDataset(AitexFolder):
    Defect_path = AitexFolder + 'Defect_images/'
    NODefect_path = AitexFolder + 'NODefect_images/'
    Mask_path = AitexFolder + 'Mask_images/'

    NODefect_subdirectories = ['2306881-210020u', '2306894-210033u', '2311517-195063u', '2311694-1930c7u',
                               '2311694-2040n7u', '2311980-185026u', '2608691-202020u']

    Reformat_Image(Defect_path + '0094_027_05.png', 4096, 256, 'white', 'right')
    Reformat_Image(Mask_path + '0094_027_05_mask.png', 4096, 256, 'black', 'right')
    os.remove(Defect_path + '0100_025_08.png')
    FlipImage(Defect_path + '0094_027_05.png')
    FlipImage(Mask_path + '0094_027_05_mask.png')

    defect_images = os.listdir(Defect_path)
    nodefect_images = []

    for i in range(len(NODefect_subdirectories)):
        for j in os.listdir(NODefect_path + NODefect_subdirectories[i]):
            nodefect_images.append(NODefect_subdirectories[i] + '/' + j)

    random.shuffle(nodefect_images)

    train_folder = AitexFolder + 'trainset/'
    validation_folder = AitexFolder + 'validationset/'
    test_folder = AitexFolder + 'testset/'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    train_range = [0, len(nodefect_images) - int(len(nodefect_images) * 20 / 100)]  # 60% train set without defects
    validation_range = [len(nodefect_images) - int(len(nodefect_images) * 20 / 100),
                                 len(nodefect_images)]  # 20% validation set without defects

    for i in range(train_range[0], train_range[1]):
        shutil.copyfile(NODefect_path + nodefect_images[i], train_folder + nodefect_images[i].split('/')[1])
    for i in range(validation_range[0], validation_range[1]):
        shutil.copyfile(NODefect_path + nodefect_images[i], validation_folder + nodefect_images[i].split('/')[1])
    for i in defect_images:
        shutil.copyfile(Defect_path + i, test_folder + i)

    DeleteFolder(Defect_path)
    DeleteFolder(NODefect_path)
    MergeMasks(Mask_path+'0044_019_04')   # Merge and delete 0044_019_04.png masks:
    MergeMasks(Mask_path+'0097_030_03')   # Merge and delete 0097_030_03.png masks:
    BinarizeMasks(Mask_path)


def checkAitex():
    if os.path.isdir(aitex_folder):
        if (os.path.isdir(aitex_train_dir) and os.path.isdir(aitex_validation_dir) and os.path.isdir(aitex_test_dir) and os.path.isdir(aitex_mask_dir)):
            return
        else:
            info_file = Config().getInfoFile()
            b.myPrint("Preparing the AITEX dataset...", info_file)
            CreateAitexDataset(aitex_folder+'/')
    else:
        print('ERROR: Run \'./utils/get_aitex.sh\' firstly!')
        sys.exit(-1)