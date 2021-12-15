from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import torch

import backbone as b
from config import *



def compute_tsne(X, label, name, tensors):
    if tensors:
        X = X.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

    tsne_result = TSNE(n_components=2).fit_transform(X)
    outputs_dir = Config().getOutputDir()
    torch.save(tsne_result, outputs_dir + name + '.pt')
    fig = plt.figure()
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 20, label)
    fig.savefig(b.assemble_pathname(name))
    plt.close('all')