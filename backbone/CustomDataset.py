import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, masks=False):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.masks = masks
        if masks:
            self.mask_dir = mask_dir
            self.all_imgs = os.listdir(main_dir)
            self.all_mask = os.listdir(mask_dir)
            

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
                tensor_mask = self.transform(mask)
            return tensor_image, tensor_mask
        else:
            return tensor_image
