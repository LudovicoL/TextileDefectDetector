from datetime import datetime
date = datetime.now()
date = date.strftime("%Y-%m-%d_%H-%M-%S")


train_dir = './dataset/AITEX/trainset'
validation_dir = './dataset/AITEX/validationset'
test_dir = './dataset/AITEX/testset'
outputs_dir = './ouputs/'

latent_space = 25
learning_rate = 1e-3
epochs = 1

patch_size = 16
stride = 16

batch_size = 256

allprint = False

plot_extension = '.pdf'