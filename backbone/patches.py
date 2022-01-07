import torch
import torch.nn.functional as F
import torchvision
from config import ANOMALY_THRESHOLD
import backbone as b

def DivideInPatches(dataset, n_channels, size, stride, masks=False):
    if not masks:
        patches = []
        for i in dataset:
            if i.shape[0] > 1:  # if the channels are greather than 1
                temp = i.unfold(1, size, stride).unfold(2, size, stride)
                patches.append(temp[0].reshape((1, int(i.shape[1] / size), int(i.shape[2] / size), size, size)).reshape((-1, n_channels, size, size)))
            else:
                patches.append(i.unfold(1, size, stride).unfold(2, size, stride).reshape((-1, n_channels, size, size)))
        return patches
    else:
        patches = []
        mask_patches = []
        for i in dataset:
            if i[0].shape[0] > 1:  # if the channels are greather than 1
                temp = i[0].unfold(1, size, stride).unfold(2, size, stride)
                patches.append(temp[0].reshape((1, int(i[0].shape[1] / size), int(i[0].shape[2] / size), size, size)).reshape((-1, n_channels, size, size)))
                temp = i[1].unfold(1, size, stride).unfold(2, size, stride)
                mask_patches.append(temp[0].reshape((1, int(i[1].shape[1] / size), int(i[1].shape[2] / size), size, size)).reshape((-1, n_channels, size, size)))
            else:
                patches.append(i[0].unfold(1, size, stride).unfold(2, size, stride).reshape((-1, n_channels, size, size)))
                mask_patches.append(i[1].unfold(1, size, stride).unfold(2, size, stride).reshape((-1, n_channels, size, size)))
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