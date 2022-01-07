from .CustomDataset import CustomDataSet, augmentationDataset, resize
from .patches import DivideInPatches, AssemblePatches, getPosition, countAnomalies
from .VAE import VariationalAutoencoder, add_noise
from .DisplayImages import display_images, plot_couple, assemble_pathname
from .train import train
from .test import test
from .SSIM import calculate_ssim, plot_ssim_histogram
from .GMM import *
from .TSNE import *
from .KDE import *

from config import Telegram_messages

import requests
import json
import os
from torchvision import transforms

def myPrint(string, filename):
    print(string)
    filename.write(string + '\n')

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)

def telegram_bot_sendtext(bot_message):
    if Telegram_messages:
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

transform = transforms.Compose([
    transforms.ToTensor(),
    # AddGaussianNoise(0.1, 0.08)
])

