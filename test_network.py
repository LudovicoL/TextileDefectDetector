# All imports
import torch
from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import random

import os

from matplotlib import pyplot as plt

import backbone as b
from variables import *
#############################################################


# Set manual seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)


# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
directories= os.listdir(outputs_dir)
print(directories)
b.date = directories[-1]
print(date)
'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[-1],std=[2])
])


test_dataset = b.CustomDataSet(test_dir, transform=transform)

fig = plt.figure()
plt.imshow(np.transpose(test_dataset.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
fig.savefig(b.assemble_pathname('Test_dataset_image28'))

channel = test_dataset[0].shape[0]
if allprint: print('Channel: ' + str(channel))

original_height = test_dataset[0].shape[1]
if allprint: print('Original height: ' + str(original_height))

original_width = test_dataset[0].shape[2]
if allprint: print('Original width: ' + str(original_width))

if allprint: print('Image size: ' + str(original_width) + '×' + str(original_height) + '×' + str(channel))

test_temp = b.DivideInPatches(test_dataset, original_height, original_width, patch_size, stride)

test_patches = torch.Tensor(1)
torch.cat(test_temp, out=test_patches)

test_loader = DataLoader(test_patches, batch_size=batch_size)

model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
model.load_state_dict(torch.load(model_path + 'state_dict'))
train_codes = torch.load(model_path + 'train_codes.pt')

test_loss, test_x, test_x_hat, test_codes, test_means, test_logvars = b.test(model, device, test_loader, train_codes)



tensor_original = torch.cat(test_x)
tensor_original = b.AssemblePatches(tensor_original, len(test_temp), channel, original_height, original_width, patch_size, stride)
fig = plt.figure()
plt.imshow(np.transpose(tensor_original.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
fig.savefig(b.assemble_pathname('Test_original_image28'))

tensor_reconstructed = torch.cat(test_x_hat)
tensor_reconstructed = b.AssemblePatches(tensor_reconstructed, len(test_temp), channel, original_height, original_width, patch_size, stride)
fig = plt.figure()
plt.imshow(np.transpose(tensor_reconstructed.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
fig.savefig(b.assemble_pathname('Test_reconstructed_image28'))

b.plot_histogram(tensor_original.__getitem__(28)[0], tensor_reconstructed.__getitem__(28)[0], b.assemble_pathname('Histograms_image28'))

# ssim = b.ssim(tensor_original.__getitem__(28), tensor_reconstructed.__getitem__(28), val_range=255)
# print('SSIM: ' + {ssim})