# All imports
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import numpy as np

import random

from matplotlib import pyplot as plt

import backbone as bc
from variables import *
#############################################################


# Set manual seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)


# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[-1],std=[2])
])


train_dataset = bc.CustomDataSet(train_dir, transform=transform)
validation_dataset = bc.CustomDataSet(validation_dir, transform=transform)
test_dataset = bc.CustomDataSet(test_dir, transform=transform)

if allprint: print('Train dataset shape: ' + str(train_dataset.__getitem__(0).shape))

f = plt.figure()
plt.imshow(np.transpose(train_dataset.__getitem__(0).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
f.savefig(bc.assemble_pathname('_train_dataset_image0'))

f = plt.figure()
plt.imshow(np.transpose(validation_dataset.__getitem__(13).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
f.savefig(bc.assemble_pathname('_validation_dataset_image13'))

f = plt.figure()
plt.imshow(np.transpose(test_dataset.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
f.savefig(bc.assemble_pathname('_test_dataset_image28'))


channel = train_dataset[0].shape[0]
if allprint: print('Channel: ' + str(channel))

original_height = train_dataset[0].shape[1]
if allprint: print('Original height: ' + str(original_height))

original_width = train_dataset[0].shape[2]
if allprint: print('Original width: ' + str(original_width))

if allprint: print('Image size: ' + str(original_width) + '×' + str(original_height) + '×' + str(channel))


train_temp = bc.DivideInPatches(train_dataset, original_height, original_width, patch_size, stride)
validation_temp = bc.DivideInPatches(validation_dataset, original_height, original_width, patch_size, stride)
test_temp = bc.DivideInPatches(test_dataset, original_height, original_width, patch_size, stride)

if allprint: print('Number of images in trainset: ' + str(len(train_temp)))
if allprint: print('Shape of every image in trainset: ' + str(train_temp[0].shape)) # 0: batch-size # 1: canale # 2: colonne # 3: righe

train_patches = torch.Tensor(1)
torch.cat(train_temp, out=train_patches)

validation_patches = torch.Tensor(1)
torch.cat(validation_temp, out=validation_patches)

test_patches = torch.Tensor(1)
torch.cat(test_temp, out=test_patches)

if allprint: print('Train patches shape: ' + str(train_patches.shape))

f = plt.figure()
plt.imshow(train_patches[100].permute(1,2,0), cmap='gray')
f.savefig(outputs_dir+date+'_train_patches_image100' + plot_extension)

train_loader = data.DataLoader(train_patches , batch_size=batch_size)
validation_loader = data.DataLoader(validation_patches , batch_size=batch_size)
test_loader = data.DataLoader(test_patches, batch_size=batch_size)


model = bc.VariationalAutoencoder(latent_space, learning_rate).to(device)

training_loss, validation_loss, train_x, train_x_hat, validation_x, validation_x_hat, train_codes, train_means, train_logvars = bc.train(model, epochs, device, train_loader, validation_loader)
test_loss, test_x, test_x_hat, test_codes, test_means, test_logvars = bc.test(model, device, test_loader, train_codes)

tensor_original = torch.cat(test_x)
tensor_original = bc.AssemblePatches(tensor_original, len(test_temp), channel, original_height, original_width, patch_size, stride)
f = plt.figure()
plt.imshow(np.transpose(tensor_original.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
f.savefig(bc.assemble_pathname('_test_original_image28'))

tensor_reconstructed = torch.cat(test_x_hat)
tensor_reconstructed = bc.AssemblePatches(tensor_reconstructed, len(test_temp), channel, original_height, original_width, patch_size, stride)
f = plt.figure()
plt.imshow(np.transpose(tensor_reconstructed.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
f.savefig(bc.assemble_pathname('_test_reconstructed_image28'))

bc.plot_histogram(tensor_original.__getitem__(28)[0], tensor_reconstructed.__getitem__(28)[0], bc.assemble_pathname('_histograms_image28'))