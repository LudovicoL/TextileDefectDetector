import os           # To read the file
import random       # shuffle
import shutil       # To copy the file
from PIL import Image
import numpy as np


def Reformat_Image(ImageFilePath, new_width, new_height, color, offset):

    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if color == 'white':
        color = (255, 255, 255, 255)
    elif color == 'black':
        color = (0, 0, 0, 255)

    if offset == 'center':
        offset = (int(round(((new_width - width) / 2), 0)), int(round(((new_height - height) / 2), 0)))
    elif offset == 'right':
        offset = (0, 0)
    elif offset == 'left':
        offset = ((new_width - width), (new_height - height))

    background = Image.new('RGBA', (new_width, new_height), color)

    background.paste(image, offset)
    background.save(ImageFilePath)
    print("Image " + ImageFilePath + " has been resized!")

def DeleteFolder(path):
    shutil.rmtree(path)
    print(path + ' deleted!')

def MergeMasks(name):
    mask1 = Image.open(name+'_mask1.png').convert('L')
    mask2 = Image.open(name+'_mask2.png').convert('L')
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask = np.add(mask1, mask2)
    mask = Image.fromarray(mask)
    mask.save(name+'_mask.png',"PNG")
    os.remove(name+'_mask1.png')
    os.remove(name+'_mask2.png')

def BinarizeMasks(Mask_path):
    thresh = 128
    maxval = 255

    all_imgs = sorted(os.listdir(Mask_path))
    for i in all_imgs:
        im_gray = np.array(Image.open(Mask_path+i).convert('L'))
        im_bool = im_gray > thresh
        im_bin = (im_gray > thresh) * maxval
        Image.fromarray(np.uint8(im_bin)).save(Mask_path+i)

def CreateAitexDataset(AitexFolder):
    Defect_path = AitexFolder + 'Defect_images/'
    NODefect_path = AitexFolder + 'NODefect_images/'
    Mask_path = AitexFolder + 'Mask_images/'

    NODefect_subdirectories = ['2306881-210020u', '2306894-210033u', '2311517-195063u', '2311694-1930c7u',
                               '2311694-2040n7u', '2311980-185026u', '2608691-202020u']

    Reformat_Image(Defect_path + '0094_027_05.png', 4096, 256, 'white', 'right')
    Reformat_Image(Mask_path + '0094_027_05_mask.png', 4096, 256, 'black', 'right')
    os.remove(Defect_path + '0100_025_08.png')

    defect_images = os.listdir(Defect_path)
    nodefect_images = []

    for i in range(len(NODefect_subdirectories)):
        for j in os.listdir(NODefect_path + NODefect_subdirectories[i]):
            nodefect_images.append(NODefect_subdirectories[i] + '/' + j)

    random.shuffle(nodefect_images)

    train_folder = AitexFolder + 'trainset/'
    validation_folder = AitexFolder + 'validationset/'
    test_folder = AitexFolder + 'testset/'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    train_range = [0, len(nodefect_images) - int(len(nodefect_images) * 20 / 100)]  # 60% train set without defects
    validation_range = [len(nodefect_images) - int(len(nodefect_images) * 20 / 100),
                                 len(nodefect_images)]  # 20% validation set without defects

    for i in range(train_range[0], train_range[1]):
        shutil.copyfile(NODefect_path + nodefect_images[i], train_folder + nodefect_images[i].split('/')[1])
    for i in range(validation_range[0], validation_range[1]):
        shutil.copyfile(NODefect_path + nodefect_images[i], validation_folder + nodefect_images[i].split('/')[1])
    for i in defect_images:
        shutil.copyfile(Defect_path + i, test_folder + i)

    DeleteFolder(Defect_path)
    DeleteFolder(NODefect_path)

    MergeMasks(Mask_path+'0044_019_04')   # Merge and delete 0044_019_04.png masks:
    MergeMasks(Mask_path+'0097_030_03')   # Merge and delete 0097_030_03.png masks:

    BinarizeMasks(Mask_path)


random.seed(10)

path = os.getcwd()
if path[-5:] == 'utils':
    AitexFolder = '../dataset/AITEX'
else:
    AitexFolder = './dataset/AITEX'



if os.path.isdir(AitexFolder):
    CreateAitexDataset(AitexFolder+'/')
else:
    print('ERROR: Run \'get_aitex.sh\' firstly!')

