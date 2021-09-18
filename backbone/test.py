import torch
from tqdm import tqdm
import backbone as bc

from variables import *

def test(model, device, test_loader, codes):

    test_x = []
    test_x_hat = []

    test_loss = 0

    means, logvars = list(), list()
    with torch.no_grad():
        model.eval()
        for x in test_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar = model(x)
            test_loss += model.loss_function(x_hat, x, mu, logvar).item()
            # =====================log=======================
            means.append(mu.detach())
            logvars.append(logvar.detach())
            test_x.append(x)
            test_x_hat.append(x_hat)
    # ===================log========================
    codes['μ'].append(torch.cat(means))
    codes['logσ2'].append(torch.cat(logvars))
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    bc.display_images(x, x_hat, 0, outputs_dir + date + '_testphase' + str(0) + plot_extension, True)

    return test_loss, test_x, test_x_hat, codes, means, logvars