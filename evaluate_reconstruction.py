# All imports
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

import backbone as b
from config import *

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-1],std=[2])
    ])


    test_dataset = b.CustomDataSet(test_dir, transform=transform)
    channels = test_dataset[0].shape[0]
    original_height = test_dataset[0].shape[1]
    original_width = test_dataset[0].shape[2]

    test_x_hat = torch.load(outputs_dir + 'test_x_hat.pt')
    
    tensor_reconstructed = torch.cat(test_x_hat)
    tensor_reconstructed = b.AssemblePatches(tensor_reconstructed, len(test_dataset), channels, original_height, original_width, patch_size, stride)
    
    print('Start plotting all figures...')
    for i in range(len(test_dataset)):
        fig = plt.figure()
        plt.imshow(np.transpose(test_dataset.__getitem__(i).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Test_image'+str(i)+'original'))
        plt.clf()
        plt.imshow(np.transpose(tensor_reconstructed.__getitem__(i).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Test_image'+str(i)+'reconstructed'))
        plt.close('all')

    print('Start plotting the histogram...')
    b.plot_couple(test_dataset.__getitem__(28)[0], tensor_reconstructed.__getitem__(28)[0], b.assemble_pathname('Histograms_image28'), histogram=True)
    plt.close('all')
    

if __name__ == '__main__':
    main()