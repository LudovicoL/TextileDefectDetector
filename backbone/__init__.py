from .patches import DivideInPatches, AssemblePatches, getPosition, countAnomalies
from .VAE import VariationalAutoencoder#, add_noise
from .DisplayImages import display_images, plot_couple, assemble_pathname
from .train import train
from .test import test
from .SSIM import calculate_ssim, plot_ssim_histogram
from .GMM import *
from .TSNE import *
from .KDE import *
from .metrics import *
from .AITEX import AitexDataSet, augmentationDataset, resize, checkAitex

from config import Telegram_messages

import requests
import json
import os
import skimage.filters

def myPrint(string, filename):
    print(string)
    filename.write(string + '\n')

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)

def telegram_bot_sendtext(bot_message):
    """
    Send a notice to a Telegram chat. To use, create a file "tg.ll" in the main folder with this form:
    {
    "token": "",    <-- bot token 
    "idchat": ""    <-- your chat id
    }
    """
    with open('./tg.ll') as f:
        data = json.load(f)
    bot_token = data['token']
    bot_chatID = data['idchat']
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return str(response)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def convert_from_0_1_to_0_255(img):
    """
    Convert an image from 0/1 to 0/255
    
    Input: an image

    Output: an image
    """
    tensor = False
    if torch.is_tensor(img):
        tensor = True
        img = img.cpu().detach().numpy()
    img*=255
    img = img.astype(int)
    if tensor:
        img = torch.from_numpy(img)
    return img


def binarize(img, threshold=None):
    """
    Set to 255/1 the pixels that are major or equal than 128/0.5020
    
    Input: 
    - img: an image
    - threshold: middle value

    Output: a binarized image
    """
    if img.type() == 'torch.IntTensor':
        MAX_VALUE = 255
        MIDDLE_VALUE = 128
    else:
        MAX_VALUE = 1
        MIDDLE_VALUE = 0.5019
    if threshold is not None:
        MIDDLE_VALUE = threshold
    # img = (img >= MIDDLE_VALUE) * MAX_VALUE
    img[img >= MIDDLE_VALUE] = MAX_VALUE
    img[img < MIDDLE_VALUE] = 0
    return img


def OtsuThreshold(img):
    """
    Compute the Otsu's Threshold of an image.
    Input: Image
    Output: float value
    """
    return skimage.filters.threshold_otsu(img)



def L2_distance(img1, img2):
    img1 = img1.permute(1,2,0).squeeze(2).cpu().detach().numpy()
    img2 = img2.permute(1,2,0).squeeze(2).cpu().detach().numpy()
    A = np.zeros((img1.shape[0], img1.shape[1]))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            A[i][j] = np.sqrt(np.square(img1[i][j] - img2[i][j]))
    return A


def WeightedHO():
    weightedHO = np.zeros((patch_size, patch_size))
    weightedHO[:, :] = 0.6
    weightedHO[int(patch_size/8) : int(7*patch_size/8), int(patch_size/8) : int(7*patch_size/8)] = 0.8
    weightedHO[int(patch_size/4) : int(3*patch_size/4), int(patch_size/4) : int(3*patch_size/4)] = 1
    # print(weightedHO)
    return weightedHO

def ScoreMap(scores):
    weightedHO = b.WeightedHO()

    p_max = np.max(scores)
    p_min = np.min(scores)

    s = np.empty(len(scores))
    for i in range(len(scores)):
        s[i] = 255 * ((p_max - scores[i])/(p_max - p_min))

    fs = np.empty((len(scores), patch_size, patch_size))
    for i in range(len(scores)):
        fs[i] = weightedHO.dot(s[i])
    fs = torch.Tensor(fs)
    return fs

def AnomalyProbabilityMap(scores, fs, ssim_masks):
    fp = []
    for i in range(len(scores)):
        alpha = 0.1
        beta = 1
        max = np.add(np.multiply(ssim_masks[i], alpha), np.multiply(fs[i], beta))
        j = 0.2
        while j <= 1:
            if torch.sum(np.add(j * ssim_masks[i], (1-j) * fs[i])) > torch.sum(max):
                max = np.add(np.multiply(ssim_masks[i], j), np.multiply(fs[i], (1-j)))
                alpha = j
                beta = (1-j)
            j += 0.1
        fp.append(torch.div(np.add(np.multiply(ssim_masks[i], alpha), np.multiply(fs[i], beta)), max))
    return fp