from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import torch

import backbone as b
from config import *

    

def compute_tsne(X, label, pathname, test=False):
    name = ''
    X = X.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    tsne_result = TSNE().fit_transform(X)
    if test:
        name = 'tsne_test_result.pt'
    else:
        name = 'tsne_result.pt'
    torch.save(tsne_result, outputs_dir + name)
    fig = plt.figure()
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 20, label)
    fig.savefig(pathname)
    plt.close('all')