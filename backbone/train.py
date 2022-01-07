import torch
from tqdm import tqdm
import backbone as b


from config import *

def train(model, epochs, device, train_loader, validation_loader):
    info_file = Config().getInfoFile()

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
            for x in loop:
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
                if epoch == epochs:
                    train_x_hat.append(x_hat)
                    train_features.append(features)
            # ===================log========================
            training_loss = (train_loss * batch_size ) / len(train_loader.dataset)
            if allFigures:
                b.display_images(x, x_hat, epoch, b.assemble_pathname('Train_phase' + str(epoch)), True)
            
            # Testing on validation set
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                loop = tqdm(validation_loader)
                for x in loop:
                    x = x.to(device)
                    # ===================forward=====================
                    x_hat, mu, logvar, features = model(x)
                    validation_loss += model.loss_function(x_hat, x, mu, logvar).item()
                    # =====================log=======================
                    if epoch == epochs:
                        validation_x_hat.append(x_hat)
                        validation_features.append(features)
            # ===================log========================
            validation_loss = ( validation_loss * batch_size ) / len(validation_loader.dataset)
            if allFigures:
                b.display_images(x, x_hat, epoch, b.assemble_pathname('Validation_phase' + str(epoch)), True)
            
            b.myPrint(f'====> Epoch: {epoch} Train set average loss: {training_loss:.4f}', info_file)
            b.myPrint(f'====> Validation set average loss: {validation_loss:.4f}', info_file)
    
    return train_x_hat, validation_x_hat, train_features, validation_features
    