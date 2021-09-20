import os
from datetime import datetime

train_dir = './dataset/AITEX/trainset'
validation_dir = './dataset/AITEX/validationset'
test_dir = './dataset/AITEX/testset'
outputs_dir = './ouputs/'

if os.path.basename(__file__) == 'train_network.py':
    date = datetime.now()
    date = date.strftime("%Y-%m-%d_%H-%M-%S")
else:
    date = os.listdir(outputs_dir)[-1]

latent_space = 25
learning_rate = 1e-3
epochs = 1

patch_size = 16
stride = 16

batch_size = 256

allprint = False

plot_extension = '.pdf'