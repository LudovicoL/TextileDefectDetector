import torch
from torch import nn

class VariationalAutoencoder(nn.Module):

    def __init__(self,
                 latent_space,
                 learning_rate
                 ):
        super().__init__()

        self.latent_space = latent_space
        self.learning_rate = learning_rate

        encoder_layers = []
        encoder_layers.append(nn.Conv2d(1, 32, 3, stride=1, padding=1, dilation=1))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Flatten(start_dim=1, end_dim=3))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(4096, 100))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)


        self.fc_mu = nn.Linear(100, self.latent_space)
        self.fc_var = nn.Linear(100, self.latent_space)


        decoder_layers = []
        decoder_layers.append(nn.Linear(self.latent_space, 4096))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(ReshapeLayer(-1, 64, 8, 8))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, dilation=5))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Conv2d(32, 1, 3, stride=1, padding=1, dilation=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)



    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)
        return [mu, logvar]



    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result



    def reparameterise(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu



    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar, z



    def loss_function(self, x_hat, x, mu, logvar, β=1):
        # Reconstruction + β * KL divergence losses summed over all elements and batch
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x, reduction='sum'
        )
        #         CE = torch.sum(torch.sum(- x_hat * torch.nn.functional.log_softmax(x, -1), -1).reshape(-1) ,-1)
        #         loss = nn.CrossEntropyLoss()
        #         CE = loss(x_hat, x.long())
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        return BCE + β * KLD



    def sample(self, num_samples, device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_space)
        z = z.to(device)
        samples = self.decode(z)
        return samples



    def generate(self, input, **kwargs):
        """
        Given an input image input, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(input)[0]



    def setOptimizer(self, model):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )
        return optimizer


class ReshapeLayer(nn.Module):
    def __init__(self, *args):
        super(ReshapeLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)