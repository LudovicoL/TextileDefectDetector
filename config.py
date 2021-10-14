import os
from datetime import datetime
import sys
import torch

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)


########################################################
# Control variables

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

debugging_mode = True
outputs_dir = './outputs/'
os.makedirs(outputs_dir, exist_ok=True)

if debugging_mode:
    isTrain = False
    for i in range(len(sys.argv)):
        if 'train_network.py' in sys.argv[i]:
            date = datetime.now()
            date = date.strftime("%Y-%m-%d_%H-%M-%S")
            outputs_dir = outputs_dir + date + '/'
            os.mkdir(outputs_dir)
            isTrain = True
            break 
    if not isTrain:
        if len(list(folders_in(outputs_dir))) == 0:
            print('Run \'train_network.py\' first!')
            sys.exit(-1)
        else:
            outputs_dir = sorted(list(folders_in(outputs_dir)))[-1]
            outputs_dir = outputs_dir + '/'
else:
    for i in range(len(sys.argv)):
        if 'test_network.py' in sys.argv[i]:
            if os.path.isfile('state_dict'):
                print('Run \'train_network.py\' first!')
                sys.exit(-1)
            break

plot_extension = '.pdf'
allprint = True
allFigures = True
seed = 0

########################################################
# Datasets
# AITEX
train_dir = './dataset/AITEX/trainset'
validation_dir = './dataset/AITEX/validationset'
test_dir = './dataset/AITEX/testset'
mask_dir = './dataset/AITEX/Mask_images'

########################################################
# Network parameters

latent_space = 25
learning_rate = 1e-3
epochs = 19

patch_size = 16
stride = 16

batch_size = 64
