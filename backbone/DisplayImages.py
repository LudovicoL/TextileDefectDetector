from matplotlib import pyplot as plt
from variables import *

def assemble_pathname(filename):
    return outputs_dir + date + filename + plot_extension

def display_images(in_, out, epoch, filename, count=False):
    color_count = 'chocolate'

    f = plt.figure()
    in_pic = in_.data.cpu().view(-1, patch_size, patch_size)
    plt.style.use('grayscale')
    plt.suptitle('Epoch ' + str(epoch) + ' – real test data / reconstructions', color='k', fontsize=16)
    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.imshow(in_pic[i+4])
        plt.axis('off')
        if count:
            plt.title(str(i+1), color=color_count)
    out_pic = out.data.cpu().view(-1, patch_size, patch_size)
    for i in range(4, 8):
        plt.subplot(2,4,i+1)
        plt.imshow(out_pic[i+4])
        plt.axis('off')
        if count:
            plt.title(str(i-3), color=color_count)
    f.savefig(filename)

def plot_histogram(original, reconstructed, filename):
    fig = plt.figure(figsize=(15,5))
    columns = 2
    rows = 1
    img1 = original.cpu().detach().numpy()
    img2 = reconstructed.cpu().detach().numpy()
    fig.add_subplot(rows, columns, 1)
    plt.hist(img1)
    plt.title("Original")
    fig.add_subplot(rows, columns, 2)
    plt.hist(img2)
    plt.title("Reconstructed")
    fig.savefig(filename)
