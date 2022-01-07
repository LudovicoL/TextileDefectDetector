# All imports
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random
import time
import argparse
import os
import pickle
import sys
import cv2 as cv

import backbone as b
from config import *
#############################################################
## Argparse declaration ##
ap = argparse.ArgumentParser()
ap.add_argument("-ls", "--load_state", action="store_true", help="Load last trained model.")
ap.add_argument("-e", "--epochs", required=False, default=300, help="Number of epochs.")
ap.add_argument("-gt", "--gmm", action="store_true", help="True to use pytorch implementation, False to use sklearn library.")
ap.add_argument("--bic", action="store_true", help="True to calculate BIC.")
ap.add_argument("-sr", "--sr", action="store_true", help="True to save all reconstructed images.")
args = vars(ap.parse_args())

load_state = args["load_state"]
epochs = int(args["epochs"])
gmm_torch = args["gmm"]
compute_bic = args["bic"]
show_reconstructed = args["sr"]

# Set the output folder
outputs_dir = './outputs/'
os.makedirs(outputs_dir, exist_ok=True)
if debugging_mode and not load_state:
    outputs_dir = outputs_dir + date + '/'
    os.mkdir(outputs_dir)
elif debugging_mode and load_state:
    folders = sorted(list(b.folders_in(outputs_dir)))
    if folders[-1] == './outputs/images':
        if len(folders) > 1:
            outputs_dir = folders[-2] + '/'
        else:
            print('Error: no existing VAE model!')
            sys.exit(-1)
    else:
        outputs_dir = folders[-1] + '/'
os.makedirs(outputs_dir + 'images/', exist_ok=True)
Config().setOutputDir(outputs_dir)



def main(info_file):
    # Set manual seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    b.myPrint('Batch size: ' + str(batch_size) + "\nEpochs: " + str(epochs) + '\nLatent space: ' + str(latent_space) + '\nPatches size: ' + str(patch_size), info_file)


    train_dataset = b.CustomDataSet(train_dir, transform=b.transform)
    validation_dataset = b.CustomDataSet(validation_dir, transform=b.transform)

    channels = train_dataset[0].shape[0]
    original_height = train_dataset[0].shape[1]
    original_width = train_dataset[0].shape[2]
    train_dataset, train_widths = b.resize(train_dataset, original_width, original_height)
    validation_dataset, validation_widths = b.resize(validation_dataset, original_width, original_height)
    train_dataset, train_widths = b.augmentationDataset(train_dataset)

    b.myPrint('There are ' + str(len(train_dataset)) + ' train images with size [W]' + str(original_width) + ' × [H]' + str(original_height) + ' × [C]' + str(channels), info_file)
    b.myPrint('There are ' + str(len(validation_dataset)) + ' validation images with size [W]' + str(original_width) + ' × [H]' + str(original_height) + ' × [C]' + str(channels), info_file)

    train_patches = b.DivideInPatches(train_dataset, channels, patch_size, stride)
    validation_patches = b.DivideInPatches(validation_dataset, channels, patch_size, stride)

    train_patches = torch.cat(train_patches, dim=0).reshape(-1, channels, patch_size, patch_size)
    validation_patches = torch.cat(validation_patches, dim=0).reshape(-1, channels, patch_size, patch_size)

    b.myPrint('Number of patches in trainset: ' + str(train_patches.shape[0]), info_file)
    b.myPrint('Number of patches in validationset: ' + str(validation_patches.shape[0]), info_file)

    train_loader = DataLoader(train_patches , batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_patches , batch_size=batch_size, shuffle=False)
    
    if not load_state:
        model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
        train_x_hat, validation_x_hat, train_features, validation_features = b.train(model, epochs, device, train_loader, validation_loader)
        
        train_x_hat = torch.cat(train_x_hat, dim=0).reshape(-1, channels, patch_size, patch_size)
        validation_x_hat = torch.cat(validation_x_hat, dim=0).reshape(-1, channels, patch_size, patch_size)
        train_features = torch.cat(train_features, dim=0)
        validation_features = torch.cat(validation_features, dim=0)
        torch.save(model.state_dict(), outputs_dir + 'vae_state_dict.pt')
        torch.save(train_x_hat, outputs_dir + 'train_x_hat.pt')
        torch.save(validation_x_hat, outputs_dir + 'validation_x_hat.pt')
        torch.save(train_features, outputs_dir + 'train_features.pt')
        torch.save(validation_features, outputs_dir + 'validation_features.pt')

        print(validation_widths[0])
        tensor_reconstructed = b.AssemblePatches(validation_x_hat[:validation_widths[0]], 1, channels, original_height, validation_widths[0], patch_size, stride)
        torchvision.utils.save_image(tensor_reconstructed.__getitem__(0), b.assemble_pathname('Validation_dataset_image0_reconstructed'))
    else:
        # To avoid train phase:
        model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
        model.load_state_dict(torch.load(outputs_dir + 'vae_state_dict.pt'))
        train_x_hat = torch.load(outputs_dir + 'train_x_hat.pt')
        validation_x_hat = torch.load(outputs_dir + 'validation_x_hat.pt')
        train_features = torch.load(outputs_dir + 'train_features.pt')
        validation_features = torch.load(outputs_dir + 'validation_features.pt')
    
    if show_reconstructed:
        b.myPrint('Start saving all reconstructed images...', info_file)
        j = 0
        for i in range(len(train_dataset)):
            print(j)
            tensor_reconstructed = b.AssemblePatches(train_x_hat[j:j+train_widths[i]], 1, channels, original_height, train_widths[i], patch_size, stride)
            torchvision.utils.save_image(train_dataset.__getitem__(i)[0], b.assemble_pathname('Train_image'+str(i)+'original'))
            torchvision.utils.save_image(tensor_reconstructed.__getitem__(0), b.assemble_pathname('Train_image'+str(i)+'reconstructed'))
            j += train_widths[i]
    

    b.myPrint('Start computing SSIM for trainset...', info_file)
    
    train_ssim = []
    for i in range(len(train_patches)):
        score, _ = b.calculate_ssim(train_patches[i], train_x_hat[i])
        train_ssim.append(score)
    torch.save(train_ssim, outputs_dir + 'train_ssim.pt')
    b.plot_ssim_histogram(train_ssim, b.assemble_pathname('SSIM_Train'), 'SSIM for trainset')           # Plot train SSIM histogram
    b.myPrint('...start computing SSIM for validationset...', info_file)
    
    validation_ssim = []
    validation_ssim_masks = []
    for i in range(len(validation_patches)):
        score, diff = b.calculate_ssim(validation_patches[i], validation_x_hat[i])
        validation_ssim.append(score)
        # diff, _ = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)      # Otsu's thresholding
        validation_ssim_masks.append(diff)
    torch.save(validation_ssim, outputs_dir + 'validation_ssim.pt')
    torch.save(validation_ssim_masks, outputs_dir + 'validation_ssim_masks.pt')
    b.plot_ssim_histogram(validation_ssim, b.assemble_pathname('SSIM_Validation'), 'SSIM for validationset')    # Plot validation SSIM histogram
    b.myPrint('...end computing SSIM.', info_file)
    
    
    train_gmm = train_features.view(-1, latent_space)
    if not gmm_torch:
        train_gmm = train_gmm.cpu().detach().numpy()
    if compute_bic:
        b.myPrint('Start computing GMM with BIC...', info_file)
        gmm, gmm_n_components, gmm_covariance = b.calculateBestGMM(x = train_gmm, n_features = latent_space, device = device, gmm_torch = gmm_torch)
        torch.save(gmm_n_components, outputs_dir + 'gmm_n_components.pt')
        torch.save(gmm_covariance, outputs_dir + 'gmm_covariance.pt')
    else:
        b.myPrint('Start computing GMM...', info_file)
        gmm = b.GMM(gmm_torch, GMM_N_COMPONENTS, GMM_COVARIANCE, latent_space, device)
        torch.save(GMM_N_COMPONENTS, outputs_dir + 'gmm_n_components.pt')
        torch.save(GMM_COVARIANCE, outputs_dir + 'gmm_covariance.pt')
    gmm.fit(train_gmm)
    if not gmm_torch:
        pickle.dump(gmm, open(outputs_dir + 'gmm_model.sav', 'wb'))
    else:
        torch.save(gmm.state_dict(), outputs_dir + 'gmm_model.pt')
    train_gmm_labels = gmm.predict(train_gmm)
    torch.save(train_gmm, outputs_dir + 'train_gmm.pt')
    torch.save(train_gmm_labels, outputs_dir + 'train_gmm_labels.pt')
    
    validation_gmm = validation_features.view(-1, latent_space)
    if not gmm_torch:
        validation_gmm = validation_gmm.cpu().detach().numpy()
    validation_gmm_labels = gmm.predict(validation_gmm)
    torch.save(validation_gmm, outputs_dir + 'validation_gmm.pt')
    torch.save(validation_gmm_labels, outputs_dir + 'validation_gmm_labels.pt')
    validation_scores = np.asarray(gmm.score_samples(validation_gmm))
    '''
    print('Start computing GMM...')
    gmm = pickle.load(open(outputs_dir + 'gmm_model.sav', 'rb'))
    validation_gmm = torch.load(outputs_dir + 'validation_gmm.pt')
    validation_scores = np.asarray(gmm.score_samples(validation_gmm))
    '''
    p_max = np.max(validation_scores)
    p_min = np.min(validation_scores)

    print(p_max, p_min)
    fp = []
    for i in range(len(validation_scores)):
        validation_s = 255 * ((p_max - validation_scores[i])/(p_max - p_min))
        alpha = 0.1
        beta = 1
        max = np.sum(alpha * validation_ssim_masks[i] + beta * validation_s)
        j = 0.2
        while j <= 1:
            if (np.sum(j * validation_ssim_masks[i] + (1-j) * validation_s)) > max:
                max = np.sum(j * validation_ssim_masks[i] + (1-j) * validation_s)
                alpha = j
                beta = (1-j)
            j += 0.1
        fp.append((alpha * validation_ssim_masks[i] + beta * validation_s)/(max))
    np.savetxt(outputs_dir + 'fp.txt', np.asarray(fp).reshape(-1, 256))
    print(np.max(fp))
    torch.save(np.max(fp), outputs_dir + 'segmantion_map_threshold.pt')

    b.myPrint('...end computing GMM.', info_file)
    
    if allFigures:
        b.myPrint('Start computing t-SNE for trainset...', info_file)
        b.compute_tsne(train_gmm, train_gmm_labels, 'train_tsne', gmm_torch)
        b.myPrint('...start computing t-SNE for validationset...', info_file)
        b.compute_tsne(validation_gmm, validation_gmm_labels, 'validation_tsne', gmm_torch)
        b.myPrint('...end computing t-SNE.', info_file)
    



if __name__ == '__main__':
    start_time = time.time()
    info_file = open(outputs_dir + "info.txt", "a")
    Config().setInfoFile(info_file)
    info_file.write("-----TRAINING PHASE-----\n" + str(date) + "\n")
    main(info_file)
    b.myPrint("--- %s seconds ---\n" % (time.time() - start_time), info_file)
    b.telegram_bot_sendtext("Training finished.")
    info_file.close()