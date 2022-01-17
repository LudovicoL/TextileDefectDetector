# All imports
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random
from matplotlib import pyplot as plt
import time
import argparse
import os
import sys
import pickle
import cv2 as cv

import backbone as b
from config import *
#############################################################
## Argparse declaration ##
ap = argparse.ArgumentParser()
ap.add_argument("-gt", "--gmm", action="store_true", help="True to use pytorch implementation, False to use sklearn library.")
ap.add_argument("-sr", "--sr", action="store_true", help="True to save all reconstructed images.")
ap.add_argument("-d", "--dataset", default="aitex", help="Choose the dataset.")
args = vars(ap.parse_args())

gmm_torch = args["gmm"]
show_reconstructed = args["sr"]
dataset = args["dataset"]


# Set the output folder
outputs_dir = './outputs/'
if debugging_mode:
    if len(list(b.folders_in(outputs_dir))) == 0:
        print('Run \'train_network.py\' first!')
        sys.exit(-1)
    else:
        folders = sorted(list(b.folders_in(outputs_dir)))
        if folders[-1] == './outputs/images':
            if len(folders) > 1:
                outputs_dir = folders[-2] + '/'
            else:
                print('Run \'train_network.py\' first!')
                sys.exit(-1)
        else:
            outputs_dir = folders[-1] + '/'
else:
    if not os.path.isfile(outputs_dir + 'vae_state_dict.pt'):
        print('Run \'train_network.py\' first!')
        sys.exit(-1)
Config().setOutputDir(outputs_dir)

def main(info_file):
    # Set manual seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if dataset == "aitex":
        test_dataset = b.AitexDataSet(aitex_test_dir, masks=True)
        channels = test_dataset[0][0].shape[0]
        original_height = test_dataset[0][0].shape[1]
        original_width = test_dataset[0][0].shape[2]
        test_dataset, test_widths = b.resize(test_dataset, original_width, original_height, True)

    b.myPrint('There are ' + str(len(test_dataset)) + ' test images with size [W]' + str(original_width) + ' × [H]' + str(original_height) + ' × [C]' + str(channels), info_file)
    number_of_original_images = len(test_dataset)

    test_patches, mask_test_patches = b.DivideInPatches(test_dataset, channels, patch_size, stride, True)
    

    test_patches = torch.cat(test_patches, dim=0).reshape(-1, channels, patch_size, patch_size)
    number_of_patches = len(test_patches)
    b.myPrint('Number of patches: ' + str(number_of_patches), info_file)
    
    mask_test_patches = torch.cat(mask_test_patches, dim=0).reshape(-1, channels, patch_size, patch_size)
    torch.save(mask_test_patches, outputs_dir + 'mask_test_patches.pt')
    anomalies_number, defective = b.countAnomalies(mask_test_patches)
    b.myPrint('Number of patches with defect: ' + str(anomalies_number) + '/' + str(len(mask_test_patches)), info_file)
    
    # position = b.getPosition(number_of_original_images, original_height, original_width, patch_size)
    
    test_loader = DataLoader(test_patches, batch_size=batch_size)

    model = b.VariationalAutoencoder(latent_space, learning_rate, channels).to(device)
    model.load_state_dict(torch.load(outputs_dir + 'vae_state_dict.pt'))
    
    test_x_hat, test_features = b.test(model, device, test_loader)
    torch.save(test_x_hat, outputs_dir + 'test_x_hat.pt')

    test_x_hat = torch.cat(test_x_hat, dim=0).reshape(-1, channels, patch_size, patch_size)
    test_features = torch.cat(test_features, dim=0)
    
    torch.set_printoptions(profile="full")

    
    b.myPrint('Start computing SSIM for testset...', info_file)
    test_ssim = []
    test_ssim_masks = []
    for i in range(number_of_patches):
        score, diff = b.calculate_ssim(test_patches[i], test_x_hat[i])
        test_ssim.append(score)
        test_ssim_masks.append(diff)
    torch.save(test_ssim, outputs_dir + 'test_ssim.pt')
    torch.save(test_ssim_masks, outputs_dir + 'test_ssim_masks.pt')
    test_ssim_masks_tensor = torch.Tensor(test_ssim_masks).unsqueeze(3).permute(0, 3, 1, 2)
    b.plot_ssim_histogram(test_ssim, b.assemble_pathname('SSIM_Test'), 'SSIM for testset')      # Plot test SSIM histogram
    
    
    print('Start saving all SSIM masks...')
    max_intensity = torch.load(outputs_dir + 'max_intensity.pt')
    j = 0
    for i in range(number_of_original_images):
        tensor_reconstructed = b.AssemblePatches(test_ssim_masks_tensor[j:j+test_widths[i]], 1, channels, original_height, test_widths[i], patch_size, stride).__getitem__(0)
        tensor_reconstructed[tensor_reconstructed < max_intensity] = 0
        # otsu = b.OtsuThreshold(tensor_reconstructed[0].cpu().detach().numpy())
        # tensor_reconstructed = b.binarize(tensor_reconstructed, threshold=otsu)#.type(torch.FloatTensor)
        torchvision.utils.save_image(tensor_reconstructed, b.assemble_pathname('SSIM_Mask_'+str(i)))
        j += test_widths[i]


    # Plot SSIM with anomalies in red and normal in green
    normal_ssim = []
    anomalies_ssim = []
    for i in range(number_of_patches):
        if defective[i]:
            anomalies_ssim.append(test_ssim[i])
        else:
            normal_ssim.append(test_ssim[i])
    random.shuffle(normal_ssim)
    normal_ssim = normal_ssim[0:anomalies_number]
    fig = plt.figure()
    plt.xlim([-1, 1])
    fig.suptitle('SSIM with anomalies', fontsize=20)
    plt.hist(normal_ssim, color='green', alpha=0.5, label='Normal')
    plt.hist(anomalies_ssim, color='red', alpha=0.5, label='Anomalies')
    plt.legend(loc='upper left')
    fig.savefig(b.assemble_pathname('SSIM_Test with anomalies'))
    plt.close('all')
    # b.KDE(normal_ssim)
    b.myPrint('...end computing SSIM.', info_file)
    
    
    
    b.myPrint('Start computing GMM...', info_file)
    test_gmm = test_features.view(-1, latent_space)
    gmm_n_components = torch.load(outputs_dir + 'gmm_n_components.pt')
    gmm_covariance = torch.load(outputs_dir + 'gmm_covariance.pt')
    if not gmm_torch:
        test_gmm = test_gmm.cpu().detach().numpy()
        gmm = b.GMM(gmm_torch, gmm_n_components, gmm_covariance)
        gmm = pickle.load(open(outputs_dir + 'gmm_model.sav', 'rb'))
    else:
        gmm = b.GMM(gmm_torch, gmm_n_components, gmm_covariance, latent_space, device)
        gmm.load_state_dict(torch.load(outputs_dir + 'gmm_model.py'))
    test_gmm_labels = gmm.predict(test_gmm)
    # means = gmm.means_
    # print(means.shape)
    # variance = gmm.covariances_
    # print(variance.shape)
    # proba = gmm.predict_proba(test_gmm)
    # print(proba.shape)
    # print(proba[1])

    

    torch.save(test_gmm_labels, outputs_dir + 'test_gmm_labels.pt')
    # np.savetxt(outputs_dir + 'test_gmm.txt', test_gmm.cpu().detach().numpy())
    # np.savetxt(outputs_dir + 'test_gmm_labels.txt', test_gmm_labels.cpu().detach().numpy())
    b.myPrint('...end computing GMM.', info_file)

    if allFigures:
        b.myPrint('Start computing t-SNE...', info_file)
        b.compute_tsne(test_gmm[test_widths[0]:test_widths[0]+test_widths[1]], test_gmm_labels[test_widths[0]:test_widths[0]+test_widths[1]], 'test_tsne', gmm_torch)
        b.myPrint('...end computing t-SNE.', info_file)
    
    if show_reconstructed:
        b.myPrint('Start saving all reconstructed images...', info_file)
        j = 0
        for i in range(number_of_original_images):
            tensor_reconstructed = b.AssemblePatches(test_x_hat[j:j+test_widths[i]], 1, channels, original_height, test_widths[i], patch_size, stride).__getitem__(0)
            torchvision.utils.save_image(test_dataset.__getitem__(i)[0], b.assemble_pathname('Test_image'+str(i)+'original'))
            torchvision.utils.save_image(tensor_reconstructed, b.assemble_pathname('Test_image'+str(i)+'reconstructed'))
            j += test_widths[i]
      
        
    test_scores = np.asarray(gmm.score_samples(test_gmm))

    fs = b.ScoreMap(test_scores)        # Score map
    
    j = 0
    for i in range(number_of_original_images):
        tensor_reconstructed = b.AssemblePatches(fs[j:j+test_widths[i]], 1, channels, original_height, test_widths[i], patch_size, stride).__getitem__(0)
        torchvision.utils.save_image(tensor_reconstructed, b.assemble_pathname('FS_'+str(i)))
        j += test_widths[i]

    
    fp = b.AnomalyProbabilityMap(test_scores, fs, test_ssim_masks)

    segmantion_map_threshold = torch.load(outputs_dir + 'segmantion_map_threshold.pt')
    print('Threshold: ' + str(segmantion_map_threshold))
    
    S = []

    for k in fp:
        for i in range(patch_size):
            for j in range(patch_size):
                if k[i][j] >= segmantion_map_threshold:
                    S.append(1)
                else:
                    S.append(0)
    S = np.asarray(S).reshape(-1, patch_size, patch_size)
    
    np.savetxt(outputs_dir + 'S.txt', S.reshape(-1, 256))

    Normal = []
    Anomalies = []
    TP = []
    TN = []
    FP = []
    FN = []

    for i in range(number_of_patches):
        # difference = iou(test_ssim_diff[i], mask_test_patches[i])
        # difference = torch.abs(torch.sum(torch.subtract(test_ssim_diff[i], mask_test_patches[i]))).item()
        # if difference < THRESHOLD:
        # if test_ssim[i] < 0.75:
        # if scores[i] <= thresh:
        if np.sum(S[i]) > 0:
            Anomalies.append(i)
            if int(torch.sum(mask_test_patches[i])) > 0:
                TP.append(i)
            else:
                FP.append(i)
        else:
            Normal.append(i)
            if int(torch.sum(mask_test_patches[i])) == 0:
                TN.append(i)
            else:
                FN.append(i)

    print(len(TP), len(FP), len(TN), len(FN))
    precision = b.precision(len(TP), len(FP))
    sensitivity = b.sensitivity(len(TP), len(FN))
    b.FPR(len(FP), len(TN))
    b.F1_score (precision, sensitivity)

    # b.myPrint('TP (Anomalies): ' + str(len(TP)) + '/' + str(anomalies_number), info_file)
    # b.myPrint('FP (Anomalies): ' + str(len(FP)), info_file)
    # b.myPrint('TN (Normal): ' + str(len(TN))+ '/' + str(number_of_patches - anomalies_number), info_file)
    # b.myPrint('FN (Normal): ' + str(len(FN)), info_file)
    # b.myPrint('Precision: ' + str(precision), info_file)
    # b.myPrint('Sensitivity: ' + str(sensitivity), info_file)
    # b.myPrint('FPR: ' + str(FPR), info_file)
    # b.myPrint('TPR: ' + str(TPR), info_file)
    # b.myPrint('F1_score: ' + str(F1_score), info_file)
    

if __name__ == '__main__':
    start_time = time.time()
    info_file = open(outputs_dir + "info.txt", "a")
    Config().setInfoFile(info_file)
    info_file.write("\n-----TESTING PHASE-----\n" + str(date) + "\n")
    main(info_file)
    b.myPrint("--- %s seconds ---" % (time.time() - start_time), info_file)
    if Telegram_messages: b.telegram_bot_sendtext("Testing finished.")
    info_file.close()