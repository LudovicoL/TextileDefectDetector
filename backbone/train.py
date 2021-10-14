import torch
from tqdm import tqdm
import backbone as b

from config import *

def train(model, epochs, device, train_loader, validation_loader):

    train_x_hat = []
    validation_x_hat = []
    train_features = []
    validation_features = []


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
                x_hat, mu, logvar, features = model(x)
                loss = model.loss_function(x_hat, x, mu, logvar)
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss)
                # if epoch == epochs and batch_idx == len(train_loader)-1:
                if epoch == epochs:
                    train_x_hat.append(x_hat)
                    train_features.append(features)
            # ===================log========================
            training_loss = train_loss / len(train_loader.dataset)
            print(f'====> Epoch: {epoch} Average loss: {training_loss:.4f}')
            if allFigures:
                b.display_images(x, x_hat, epoch, b.assemble_pathname('Trainphase' + str(epoch)), True)

            # Testing on validation set
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                loopv = tqdm(validation_loader)
                for vbatch_idx, x in enumerate(loopv):
                    x = x.to(device)
                    # ===================forward=====================
                    x_hat, mu, logvar, features = model(x)
                    validation_loss += model.loss_function(x_hat, x, mu, logvar).item()
                    # =====================log=======================
                    # if epoch == epochs and vbatch_idx == len(validation_loader)-1:
                    if epoch == epochs:
                        validation_x_hat.append(x_hat)
                        validation_features.append(features)
            # ===================log========================
            validation_loss /= len(validation_loader.dataset)
            print(f'====> Validation set loss: {validation_loss:.4f}')
            if allFigures:
                b.display_images(x, x_hat, epoch, b.assemble_pathname('Validationphase' + str(epoch)), True)
    
    return training_loss, validation_loss, train_x_hat, validation_x_hat, train_features, validation_features