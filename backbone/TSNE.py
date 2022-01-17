from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
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
    plt.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], s=20, c=label, cmap='Set1')
    fig.savefig(b.assemble_pathname(name))
    plt.close('all')