from skimage.metrics import structural_similarity as ssim
import numpy as np
from matplotlib import pyplot as plt

def calculate_ssim(img1, img2, tensors=True):
    if tensors:
        img1 = img1.permute(1,2,0).cpu().detach().numpy().squeeze(2)
        img2 = img2.permute(1,2,0).cpu().detach().numpy().squeeze(2)
    score, diff = ssim(img1, img2, full=True)
    diff = 1 - diff
    return score, diff

def plot_ssim_histogram(ssim, filename, title):
    fig = plt.figure()
    plt.xlim([-1, 1])
    fig.suptitle(title, fontsize=20)
    plt.hist(ssim, color='green')
    fig.savefig(filename)
    plt.close('all')