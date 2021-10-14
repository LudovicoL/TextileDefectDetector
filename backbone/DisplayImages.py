from matplotlib import pyplot as plt
from config import *

def assemble_pathname(filename):
    return outputs_dir + filename + plot_extension

def display_images(in_, out, epoch, filename, count=False):
    color_count = 'chocolate'

    min_range = min(batch_size, 4)

    fig = plt.figure()
    in_pic = in_.data.cpu().view(-1, patch_size, patch_size)
    plt.style.use('grayscale')
    plt.suptitle('Epoch ' + str(epoch) + ' – real test data / reconstructions', color='k', fontsize=16)
    for i in range(min_range):
        plt.subplot(2,min_range,i+1)
        plt.imshow(in_pic[i])
        plt.axis('off')
        if count:
            plt.title(str(i+1), color=color_count)
    out_pic = out.data.cpu().view(-1, patch_size, patch_size)
    for i in range(min_range):
        plt.subplot(2,min_range,i+1+min_range)
        plt.imshow(out_pic[i])
        plt.axis('off')
        if count:
            plt.title(str(i+1), color=color_count)
    fig.savefig(filename)
    plt.close('all')

def plot_couple(original, reconstructed, filename, histogram=False):
    fig = plt.figure(figsize=(15,5))
    columns = 2
    rows = 1
    img1 = original.cpu().detach().numpy()
    img2 = reconstructed.cpu().detach().numpy()
    fig.add_subplot(rows, columns, 1)
    plt.hist(img1) if histogram else plt.imshow(img1)       # plt.hist(img1, range=[0, 1])
    plt.title("Original")
    fig.add_subplot(rows, columns, 2)
    plt.hist(img2) if histogram else plt.imshow(img2)       # plt.hist(img2, range=[0, 1])
    plt.title("Reconstructed")
    fig.savefig(filename)
    plt.close('all')
