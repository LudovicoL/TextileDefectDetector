import torch
from tqdm import tqdm
import backbone as b

from variables import *

def train(model, epochs, device, train_loader, validation_loader):

    train_x = []
    train_x_hat = []
    validation_x = []
    validation_x_hat = []

    codes = dict(μ=list(), logσ2=list())
    training_loss = 0
    validation_loss = 0
    for epoch in range(0, epochs + 1):
        # Training
        if epoch > 0:  # test untrained net first
            model.train()
            train_loss = 0
            loop = tqdm(train_loader)
            optimizer = model.setOptimizer(model)
            for batch_idx, x in enumerate(loop):
                x = x.to(device)
                # ===================forward=====================
                x_hat, mu, logvar = model(x)
                loss = model.loss_function(x_hat, x, mu, logvar)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss)
                if epoch == epochs - 1:
                    train_x.append(x)
                    train_x_hat.append(x_hat)
            # ===================log========================
            training_loss = train_loss / len(train_loader.dataset)
            print(f'====> Epoch: {epoch} Average loss: {training_loss:.4f}')
            b.display_images(x, x_hat, epoch, b.assemble_pathname('Trainphase' + str(epoch)), True)

            # Testing on validation set
            means, logvars = list(), list()
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                for x in validation_loader:
                    x = x.to(device)
                    # ===================forward=====================
                    x_hat, mu, logvar = model(x)
                    validation_loss += model.loss_function(x_hat, x, mu, logvar).item()
                    # =====================log=======================
                    means.append(mu.detach())
                    logvars.append(logvar.detach())
                    if epoch == epochs - 1:
                        validation_x.append(x)
                        validation_x_hat.append(x_hat)
            # ===================log========================
            codes['μ'].append(torch.cat(means))
            codes['logσ2'].append(torch.cat(logvars))
            validation_loss /= len(validation_loader.dataset)
            print(f'====> Validation set loss: {validation_loss:.4f}')
            b.display_images(x, x_hat, epoch, b.assemble_pathname('Validationphase' + str(epoch)), True)
    
    return training_loss, validation_loss, train_x, train_x_hat, validation_x, validation_x_hat, codes, means, logvars