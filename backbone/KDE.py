from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt

def KDE(x):
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(x)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x)

    plt.fill_between(x, np.exp(logprob), alpha=0.5)
    plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
