import torch.nn.functional as F
from config import allprint

def DivideInPatches(dataset, original_height, original_width, size, stride):
    patches = []
    for i in dataset:
        if i.shape[0] > 1:  # if the channels are greather than 1
            temp = i.unfold(1, size, stride).unfold(2, size, stride)
            patches.append(temp[0].reshape((1, int(original_height / size), int(original_width / size), size, size)).reshape((-1, 1, size, size)))
        else:
            patches.append(i.unfold(1, size, stride).unfold(2, size, stride).reshape((-1, 1, size, size)))
    return patches


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