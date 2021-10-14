# All imports
import torch
from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import random

from matplotlib import pyplot as plt

import backbone as b
from config import *
#############################################################

def main():
    # Set manual seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-1],std=[2])
    ])


    test_dataset = b.CustomDataSet(test_dir, transform=transform)

    if allFigures:
        fig = plt.figure()
        plt.imshow(np.transpose(test_dataset.__getitem__(28).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Test_dataset_image28'))
        plt.close('all')

    channels = test_dataset[0].shape[0]
    # if allprint: print('Channels: ' + str(channels))

    original_height = test_dataset[0].shape[1]
    # if allprint: print('Original height: ' + str(original_height))

    original_width = test_dataset[0].shape[2]
    # if allprint: print('Original width: ' + str(original_width))

    if allprint: print('Images size: [W]' + str(original_width) + ' × [H]' + str(original_height) + ' × [C]' + str(channels))

    test_temp = b.DivideInPatches(test_dataset, original_height, original_width, patch_size, stride)

    # test_patches = torch.Tensor(1)
    # torch.cat(test_temp, out=test_patches)
    test_patches = torch.stack(test_temp).reshape(-1, channels, patch_size, patch_size)

    test_loader = DataLoader(test_patches, batch_size=batch_size)

    model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
    model.load_state_dict(torch.load(outputs_dir + 'state_dict'))
    
    test_loss, test_x_hat, test_features = b.test(model, device, test_loader)
    torch.save(test_x_hat, outputs_dir + 'test_x_hat.pt')

    test_x_hat = torch.stack(test_x_hat).reshape(-1, channels, patch_size, patch_size)
    test_features = torch.stack(test_features)

    print('Start computing SSIM for testset...')
    test_ssim = []
    for i in range(len(test_patches)):
        test_ssim.append(b.calculate_ssim(test_patches[i], test_x_hat[i]))
    np.savetxt(outputs_dir + 'test_ssim.txt', test_ssim)
    test_ssim = torch.tensor(test_ssim)
    print('...end computing SSIM.')

    print('Start computing GMM with BIC...')
    gmm_testset = test_features.view(-1, latent_space)
    idx = torch.randperm(gmm_testset.shape[0])
    gmm_testset = gmm_testset[idx].view(gmm_testset.size())[:int(gmm_testset.shape[0]/10)]
    n_components = torch.load(outputs_dir + 'n_components.pt')
    covariance = torch.load(outputs_dir + 'covariance.pt')
    gmm = b.GaussianMixture(n_components=n_components, n_features=latent_space, covariance_type=covariance).to(device)
    gmm.load_state_dict(torch.load(outputs_dir + 'gmm_state_dict'))
    gmmTest_y = gmm.predict(gmm_testset)
    torch.save(gmmTest_y, outputs_dir + 'gmmTest_y.pt')
    # np.savetxt(outputs_dir + 'gmmTest_features.txt', gmm_testset.cpu().detach().numpy())
    # np.savetxt(outputs_dir + 'gmmTest_y.txt', gmmTest_y.cpu().detach().numpy())
    print('...end computing GMM.')

    if allFigures:
        print('Start computing t-SNE...')
        # b.compute_tsne(gmm_testset, gmmTest_y, b.assemble_pathname('tsne_test_result'))
        gmm_trainset = torch.load(outputs_dir + 'gmm_trainset.pt')
        gmm_validationset = torch.load(outputs_dir + 'gmm_validationset.pt')
        gmmTrain_y = torch.load(outputs_dir + 'gmmTrain_y.pt')
        gmmValidation_y = torch.load(outputs_dir + 'gmmValidation_y.pt')
        gmm_x = torch.Tensor(1)
        gmm_x = torch.cat((gmm_trainset, gmm_validationset, gmm_testset), dim=0)
        gmm_y = torch.Tensor(1)
        gmm_y = torch.cat((gmmTrain_y, gmmValidation_y, gmmTest_y), dim=0)
        b.compute_tsne(gmm_x, gmm_y, b.assemble_pathname('tsne_test_result'), test=True)
        print('...end computing t-SNE.')


if __name__ == '__main__':
    main()