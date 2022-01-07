import torch
import backbone as b

from config import *

def test(model, device, test_loader):
    info_file = Config().getInfoFile()
    
    test_x_hat = []
    test_features = []
    test_loss = 0

    with torch.no_grad():
        model.eval()
        for x in test_loader:
            x = x.to(device)
            # ===================forward=====================
            x_hat, mu, logvar, features = model(x)
            test_loss += model.loss_function(x_hat, x, mu, logvar).item()
            # =====================log=======================
            test_x_hat.append(x_hat)
            test_features.append(features)
    # ===================log========================
    test_loss /= len(test_loader.dataset)
    b.myPrint(f'====> Test set loss: {test_loss:.4f}', info_file)
    if allFigures:
        b.display_images(x, x_hat, 0, b.assemble_pathname('Test_phase'), True)

    return test_x_hat, test_features