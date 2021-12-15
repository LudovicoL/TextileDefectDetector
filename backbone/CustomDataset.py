import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from config import mask_dir

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
    for i in range(len(dataset)):
        j = dataset.__getitem__(i)
        ds.append(j)
        j = np.transpose(j.numpy(), (1, 2, 0))
        ds.append(torch.tensor(np.fliplr(j).copy()).permute(2, 0, 1))   # left-right flip
        ds.append(torch.tensor(np.flipud(j).copy()).permute(2, 0, 1))   # up-down flip
        blurred = gaussian_filter(j, sigma=0.5)         # blur
        ds.append(torch.tensor(blurred).permute(2, 0, 1))
    return ds
