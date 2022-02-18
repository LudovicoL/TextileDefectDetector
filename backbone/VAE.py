import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

from config import *

DENSE = 100
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
                 channels,
                 patch_size
                 ):
        super().__init__()

        self.latent_space = latent_space
        self.learning_rate = learning_rate
        self.channels = channels
        self.patch_size = patch_size

        # Encoder
        encoder_layers = []
        if RESNET == 0:
            encoder_layers.append(nn.Conv2d(self.channels, 32, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1))
            encoder_layers.append(nn.ReLU())
        else:
            encoder_layers.append(Residual(self.channels, 32, use_1x1conv=True, strides=1))
            encoder_layers.append(Residual(32, 64, use_1x1conv=True, strides=2))
            encoder_layers.append(Residual(64, 64))
            encoder_layers.append(Residual(64, 64))
        encoder_layers.append(nn.Flatten(start_dim=1, end_dim=3))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(int(self.patch_size*self.patch_size*16), DENSE))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)


        self.fc_mu = nn.Linear(DENSE, self.latent_space)
        self.fc_var = nn.Linear(DENSE, self.latent_space)
        

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.latent_space, int(self.patch_size*self.patch_size*16)))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(ReshapeLayer(-1, 64, int(self.patch_size/2), int(self.patch_size/2)))
        decoder_layers.append(nn.ReLU())
        if RESNET == 0:
            if self.patch_size == 16:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, dilation=5))
            elif self.patch_size == 32:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 7, stride=1, padding=1, dilation=3))
            elif self.patch_size == 64:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 18, stride=1, padding=1, dilation=2))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Conv2d(32, self.channels, 3, stride=1, padding=1, dilation=1))
            decoder_layers.append(nn.Sigmoid())
        elif RESNET == 1:
            if self.patch_size == 16:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, dilation=5))
            elif self.patch_size == 32:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 7, stride=1, padding=1, dilation=3))
            elif self.patch_size == 64:
                decoder_layers.append(nn.ConvTranspose2d(64, 32, 18, stride=1, padding=1, dilation=2))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=32, out_channels=self.channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(self.channels, momentum=MOMENTUM),
                    nn.Sigmoid()
                ))
        self.decoder = nn.Sequential(*decoder_layers)
            
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


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
'''
from torch.autograd import Variable

class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 latent_space,
                 learning_rate,
                 channels,
                 fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(VariationalAutoencoder, self).__init__()

        self.learning_rate = 1e-4
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        self.c0 = nn.Conv2d(3, 3, 3, stride=1, padding=1, dilation=1)
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = x.repeat(1, 3, 1, 1)
        # print(x.shape)
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(16, 16), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst, mu, logvar, z
    
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
    
    def setOptimizer(self, model):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )
        return optimizer
'''