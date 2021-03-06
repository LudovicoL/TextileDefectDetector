from datetime import datetime
import torch
    
########################################################
# Control variables

date = datetime.now()
date = date.strftime("%Y-%m-%d_%H-%M-%S")

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

debugging_mode = True

plot_extension = '.png'
allprint = False
allFigures = False
Telegram_messages = True
seed = 0

TYPE_OF_TEXTILE = '03/'

########################################################
# Datasets
# AITEX
aitex_folder = './dataset/AITEX'
aitex_train_dir = aitex_folder + '/trainset/'+TYPE_OF_TEXTILE
aitex_validation_dir = aitex_folder + '/validationset/'+TYPE_OF_TEXTILE
aitex_test_dir = aitex_folder + '/testset/'+TYPE_OF_TEXTILE
aitex_mask_dir = aitex_folder + '/Mask_images/'
CUT_PATCHES = 6

# NEW DATASET
newdataset_folder = './dataset/new_dataset'
newdataset_train_dir = newdataset_folder + '/trainset/'
newdataset_validation_dir = newdataset_folder + '/validationset/'
newdataset_test_dir = newdataset_folder + '/testset/'
newdataset_mask_dir = newdataset_folder + '/Mask_images/'

########################################################
# Network parameters

EPOCHS = 200

latent_space = 25

patch_size = 32
stride = patch_size

batch_size = 256

learning_rate = 1e-4

MOMENTUM = 0.01

ANOMALY_THRESHOLD = 2       # Ignore patches that have less than ANOMALY_THRESHOLD+1 white pixels

BIC_MAX_RANGE = 100
GMM_N_COMPONENTS = 2 #98
GMM_COVARIANCE = 'full'

########################################################
class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    def setOutputDir(self, outputs_dir):
        self.outputs_dir = outputs_dir
    def getOutputDir(self):
        return self.outputs_dir
    def setInfoFile(self, info_file):
        self.info_file = info_file
    def getInfoFile(self):
        return self.info_file