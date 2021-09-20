import os
from datetime import datetime
import sys

train_dir = './dataset/AITEX/trainset'
validation_dir = './dataset/AITEX/validationset'
test_dir = './dataset/AITEX/testset'
outputs_dir = './outputs/'

date = ''
# if os.path.basename(__file__) == 'train_network.py':
if sys.argv[0] == 'train_network.py':
    print(sys.argv[0])
    date = datetime.now()
    date = date.strftime("%Y-%m-%d_%H-%M-%S")
else:
    if len(os.listdir(outputs_dir)) == 0:
        print('Run \'train_network.py\' first!')
        sys.exit(-1)
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

model_path = outputs_dir + '/' + date + '/'