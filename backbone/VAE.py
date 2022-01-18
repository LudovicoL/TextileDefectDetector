import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

RESNET = 0

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

class VariationalAutoencoder(nn.Module):

    def __init__(self,
                 latent_space,
                 learning_rate,
                 channels
                 ):
        super().__init__()

        self.latent_space = latent_space
        self.learning_rate = learning_rate

        encoder_layers = []
        if RESNET == 0:
            encoder_layers.append(nn.Conv2d(channels, 32, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
        else:
            self.c0 = nn.Conv2d(3, 3, 3, stride=1, padding=1, dilation=1)
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-1]      # delete the last fc layer.
            self.resnet = nn.Sequential(*modules)
            encoder_layers.append(Residual(channels, 32, use_1x1conv=True, strides=1))
            encoder_layers.append(Residual(32, 64, use_1x1conv=True, strides=2))
            encoder_layers.append(Residual(64, 64))
            encoder_layers.append(Residual(64, 64))
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
        
        if RESNET == 0:
            decoder_layers.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, dilation=5))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Conv2d(32, channels, 3, stride=1, padding=1, dilation=1))
            decoder_layers.append(nn.Sigmoid())
        elif RESNET == 1:
            decoder_layers.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, dilation=5))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                ))
        else:
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(channels),
                    nn.Sigmoid()
                ))
        self.decoder = nn.Sequential(*decoder_layers)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # x = self.c0(x)
        
        if RESNET > 0:
            input = input.repeat(1, 3, 1, 1)
            input = self.resnet(input)  # ResNet
            # input = input.view(input.size(0), -1)  # flatten output of conv
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
        # x = self.decode(z)
        # print(x.shape)
        return self.decode(z), mu, logvar, z
        
        
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def loss_function(self, x_hat, x, mu, logvar):
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        return elbo



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


class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
