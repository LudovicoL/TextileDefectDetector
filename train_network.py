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

    info_file = open(outputs_dir + "info.txt", "w")
    info_file.write('Batch size: ' + str(batch_size) + "\nEpochs: " + str(epochs))
    info_file.close()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-1],std=[2])     # Normalize data between (0,1)
    ])


    train_dataset = b.CustomDataSet(train_dir, transform=transform)
    validation_dataset = b.CustomDataSet(validation_dir, transform=transform)
    

    if allprint: print('Train dataset shape: ' + str(train_dataset.__getitem__(0).shape))

    if allFigures:
        fig = plt.figure()
        plt.imshow(np.transpose(train_dataset.__getitem__(0).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Train_dataset_image0'))

        fig = plt.figure()
        plt.imshow(np.transpose(validation_dataset.__getitem__(13).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Validation_dataset_image13'))
        plt.close('all')


    channels = train_dataset[0].shape[0]
    # if allprint: print('Channels: ' + str(channels))

    original_height = train_dataset[0].shape[1]
    # if allprint: print('Original height: ' + str(original_height))

    original_width = train_dataset[0].shape[2]
    # if allprint: print('Original width: ' + str(original_width))

    if allprint: print('Images size: [W]' + str(original_width) + ' × [H]' + str(original_height) + ' × [C]' + str(channels))


    train_temp = b.DivideInPatches(train_dataset, original_height, original_width, patch_size, stride)
    validation_temp = b.DivideInPatches(validation_dataset, original_height, original_width, patch_size, stride)


    # if allprint: print('Number of images in trainset: ' + str(len(train_temp)))
    # if allprint: print('Shape of every image in trainset: ' + str(train_temp[0].shape)) # 0: batch-size # 1: canale # 2: colonne # 3: righe

    # train_patches = torch.Tensor(1)
    # torch.cat(train_temp, out=train_patches)
    train_patches = torch.stack(train_temp).reshape(-1, channels, patch_size, patch_size)
    if allprint: print('Number of images in trainset: ' + str(train_patches.shape[0]))
    if allprint: print('Shape of every image in trainset: ' + str(train_patches[0].shape)) # 0: canale # 1: colonne # 2: righe

    # validation_patches = torch.Tensor(1)
    # torch.cat(validation_temp, out=validation_patches)
    validation_patches = torch.stack(validation_temp).reshape(-1, channels, patch_size, patch_size)
    if allprint: print('Number of images in validationset: ' + str(validation_patches.shape[0]))
    if allprint: print('Shape of every image in validationset: ' + str(validation_patches[0].shape)) # 0: canale # 1: colonne # 2: righe


    if allFigures:
        fig = plt.figure()
        plt.imshow(train_patches[100].permute(1,2,0), cmap='gray')
        fig.savefig(b.assemble_pathname('Train_patches_image100'))
        plt.close('all')

    train_loader = DataLoader(train_patches , batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_patches , batch_size=batch_size, shuffle=False)
    
    
    model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
    training_loss, validation_loss, train_x_hat, validation_x_hat, train_features, validation_features = b.train(model, epochs, device, train_loader, validation_loader)
      
    train_x_hat = torch.stack(train_x_hat).reshape(-1, channels, patch_size, patch_size)
    validation_x_hat = torch.stack(validation_x_hat).reshape(-1, channels, patch_size, patch_size)
    train_features = torch.stack(train_features)
    validation_features = torch.stack(validation_features)
    torch.save(model.state_dict(), outputs_dir + 'state_dict')
    torch.save(training_loss, outputs_dir + 'training_loss.pt')
    torch.save(validation_loss, outputs_dir + 'validation_loss.pt')
    torch.save(train_x_hat, outputs_dir + 'train_x_hat.pt')
    torch.save(validation_x_hat, outputs_dir + 'validation_x_hat.pt')
    torch.save(train_features, outputs_dir + 'train_features.pt')
    torch.save(validation_features, outputs_dir + 'validation_features.pt')

    
    if allFigures:
        tensor_reconstructed = b.AssemblePatches(validation_x_hat, len(validation_dataset), channels, original_height, original_width, patch_size, stride)
        fig = plt.figure()
        plt.imshow(np.transpose(tensor_reconstructed.__getitem__(13).cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        fig.savefig(b.assemble_pathname('Validation_dataset_image13_reconstructed'))
        plt.close('all')
    
    '''
    # To avoid train phase:
    model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
    model.load_state_dict(torch.load(outputs_dir + 'state_dict'))
    train_x_hat = torch.load(outputs_dir + 'train_x_hat.pt')
    validation_x_hat = torch.load(outputs_dir + 'validation_x_hat.pt')
    train_features = torch.load(outputs_dir + 'train_features.pt')
    validation_features = torch.load(outputs_dir + 'validation_features.pt')
    '''


    
    print('Start computing SSIM for trainset...')
    b.plot_couple(train_x_hat[16].permute(1,2,0), train_x_hat[16].permute(1,2,0), b.assemble_pathname('prova'))
    train_ssim = []
    for i in range(len(train_patches)):
        train_ssim.append(b.calculate_ssim(train_patches[i], train_x_hat[i]))
    np.savetxt(outputs_dir + 'ssim.txt', train_ssim)
    train_ssim = torch.tensor(train_ssim)
    print('...start computing SSIM for validationset...')
    validation_ssim = []
    for i in range(len(validation_patches)):
        validation_ssim.append(b.calculate_ssim(validation_patches[i], validation_x_hat[i]))
    np.savetxt(outputs_dir + 'ssim.txt', validation_ssim)
    validation_ssim = torch.tensor(validation_ssim)
    if allprint:
        print("SSIM train number of elements: " + str(train_ssim.shape[0]))
        print("SSIM validation number of elements: " + str(validation_ssim.shape[0]))
    print('...end computing SSIM.')
    
    
    if allprint: print('Train features shape: ' + str(train_features.shape))

    print('Start computing GMM with BIC...')
    gmm_trainset = train_features.view(-1, latent_space)
    idx = torch.randperm(gmm_trainset.shape[0])
    gmm_trainset = gmm_trainset[idx].view(gmm_trainset.size())[:int(gmm_trainset.shape[0]/10)]
    gmm, n_components, covariance = b.calculateBestGMM(x=gmm_trainset, n_features = latent_space, device = device)
    torch.save(gmm.state_dict(), outputs_dir + 'gmm_state_dict')
    torch.save(n_components, outputs_dir + 'n_components.pt')
    torch.save(covariance, outputs_dir + 'covariance.pt')
    gmmTrain_y = gmm.predict(gmm_trainset)
    torch.save(gmm_trainset, outputs_dir + 'gmm_trainset.pt')
    torch.save(gmmTrain_y, outputs_dir + 'gmmTrain_y.pt')

    gmm_validationset = validation_features.view(-1, latent_space)
    idx = torch.randperm(gmm_validationset.shape[0])
    gmm_validationset = gmm_validationset[idx].view(gmm_validationset.size())[:int(gmm_validationset.shape[0]/10)]
    gmmValidation_y = gmm.predict(gmm_validationset)
    torch.save(gmm_validationset, outputs_dir + 'gmm_validationset.pt')
    torch.save(gmmValidation_y, outputs_dir + 'gmmValidation_y.pt')
    print('...end computing GMM.')

    if allFigures:
        print('Start computing t-SNE...')
        gmm_x = torch.Tensor(1)
        gmm_x = torch.cat((gmm_trainset, gmm_validationset), dim=0)
        gmm_y = torch.Tensor(1)
        gmm_y = torch.cat((gmmTrain_y, gmmValidation_y), dim=0)
        b.compute_tsne(gmm_x, gmm_y, b.assemble_pathname('tsne_train_result'))
        print('...end computing t-SNE.')
   


if __name__ == '__main__':
    main()
