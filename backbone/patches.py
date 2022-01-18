import torch
import torch.nn.functional as F
import torchvision
from config import ANOMALY_THRESHOLD
import backbone as b

def DivideInPatches(dataset, n_channels, size, stride, masks=False):
    if not masks:
        patches = []
        for i in dataset:
            p = i.unfold(1, size, stride).unfold(2, size, stride)
            p = p.contiguous().view(p.size(0), -1, size, size).permute(1,0,2,3)
            patches.append(p)
        return patches
    else:
        patches = []
        mask_patches = []
        for i in dataset:
            p = i[0].unfold(1, size, stride).unfold(2, size, stride)
            p = p.contiguous().view(p.size(0), -1, size, size).permute(1,0,2,3)
            patches.append(p)
            m = i[1].unfold(1, size, stride).unfold(2, size, stride)
            m = p.contiguous().view(m.size(0), -1, size, size).permute(1,0,2,3)
            mask_patches.append(m)
        return patches, mask_patches


def AssemblePatches(patches_tensor, number_of_images, channel, height, width, patch_size, stride):
    temp = patches_tensor.contiguous().view(number_of_images, channel, -1, patch_size*patch_size)
    # print(temp.shape) # [number_of_images, C, number_patches_all, patch_size*patch_size]
    temp = temp.permute(0, 1, 3, 2) 
    # print(temp.shape) # [number_of_images, C, patch_size*patch_size, number_patches_all]
    temp = temp.contiguous().view(number_of_images, channel*patch_size*patch_size, -1)
    # print(temp.shape) # [number_of_images, C*prod(kernel_size), L] as expected by Fold
    output = F.fold(temp, output_size=(height, width), kernel_size=patch_size, stride=stride)
    # print(output.shape) # [number_of_images, C, H, W]
    return output

def getPosition(number_of_images, original_height, original_width, patch_size):
    positions = []
    k = int(original_width/patch_size)
    for n in range(number_of_images):
        for i in range(int(original_height/patch_size)):
            for j in range(k):
                positions.append([n, i * k + j, patch_size * j, patch_size * i])   # number of image, number of patch, x-position, y-position
    return positions

def countAnomalies(mask_test_patches, save=False):
    number_of_defects = 0
    defective = []
    for idx, item in enumerate(mask_test_patches):
        if int(torch.sum(item)) > ANOMALY_THRESHOLD:
            number_of_defects += 1
            defective.append(True)
            if save:
                torchvision.utils.save_image(mask_test_patches[idx], b.assemble_pathname('Mask_patches_image'+str(idx)))
        else:
            defective.append(False)
    return number_of_defects, defective