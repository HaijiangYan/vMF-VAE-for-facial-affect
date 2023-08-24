import torch
import torch.nn as nn
import torch.nn.functional as F
from hypersphere.distributions import VonMisesFisher
from hypersphere.distributions import HypersphericalUniform

class VAE(torch.nn.Module):
    
    def __init__(self, z_dim=3, size=(64, 40), distribution='vmf', kappa=1, radius=1, device="cuda"):
        """
        ModelVAE initializer
        :param z_dim: dimension of the latent representation
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(VAE, self).__init__()
        
        self.z_dim, self.size, self.distribution, self.device = z_dim, size, distribution, device
        self.kappa, self.radius = kappa, radius
        
        # encoder layers: 3 conv + 2 fc
        self.cov_en0 = nn.Conv2d(
        	in_channels=1, 
        	out_channels=16, 
        	kernel_size=(3, 3), 
        	stride=2, 
        	padding=1)
        self.pool_0 = nn.MaxPool2d(kernel_size=2)
        self.cov_en1 = nn.Conv2d(
        	in_channels=16, 
        	out_channels=32, 
        	kernel_size=(5, 5), 
        	padding="same")
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # self.cov_en2 = nn.Conv2d(
        #     in_channels=32, 
        #     out_channels=32, 
        #     kernel_size=(5, 5), 
        #     padding="same")

        self.fc_en3 = nn.Linear(32*int(size[0]/8)*int(size[1]/8), 64)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(64, z_dim)
            self.fc_var =  nn.Linear(64, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(64, z_dim)
            self.fc_var = nn.Linear(64, 1)
        else:
            raise NotImplemented
            
        # decoder layers
        self.fc_de0 = nn.Linear(z_dim, 64)
        self.fc_de1 = nn.Linear(64, 32*int(size[0]/8)*int(size[1]/8))

        self.Tcov_de2 = nn.ConvTranspose2d(
        	in_channels=32, 
        	out_channels=16, 
        	kernel_size=(5, 5), 
        	stride=2, 
        	padding=2, 
            output_padding=1
        	)
        self.Tcov_de3 = nn.ConvTranspose2d(
            in_channels=16, 
            out_channels=8, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1, 
            output_padding=1
            )
        self.Tcov_de4 = nn.ConvTranspose2d(
        	in_channels=8, 
        	out_channels=1, 
        	kernel_size=(3, 3), 
        	stride=2, 
        	padding=1, 
            output_padding=1
        	)

        # classifier
        self.linear = nn.Linear(z_dim, 7)
        self.softmax_class = nn.Softmax(dim=1)

        # dropout
        # self.dropout = nn.Dropout(p=0.3)  # can be muted by model.eval()

    def encoder(self, x):
        # 2 hidden layers encoder
        x = self.pool_0(F.relu(self.cov_en0(x)))
        x = self.pool_1(F.relu(self.cov_en1(x)))
        # x = F.relu(self.cov_en2(x))
        x = x.view(-1, 32*int(self.size[0]/8)*int(self.size[1]/8))
        # x = self.dropout(x)
        x = self.fc_en3(x)  # dropout should be placed after activation
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = self.radius * z_mean / z_mean.norm(dim=-1, keepdim=True)  # keep the points on sphere
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x))  # this is just the size of kappa
        else:
            raise NotImplemented
        
        return z_mean, z_var

    def classifier(self, z):

        x = self.linear(z)
        x = self.softmax_class(x)

        return x

        
    def decoder(self, z):
        
        x = F.relu(self.fc_de0(z))
        x = F.relu(self.fc_de1(x))
        x = x.view(-1, 32, int(self.size[0]/8), int(self.size[1]/8))  
        # dimensionality transform: batch, channel, size...
        x = F.relu(self.Tcov_de2(x))
        x = F.relu(self.Tcov_de3(x))
        x = F.sigmoid(self.Tcov_de4(x))
        
        return x
        
    def reparameterize(self, z_mean, z_var, radius):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var, k=self.kappa, radius=radius)  # new real kappa is here
            p_z = HypersphericalUniform(self.z_dim - 1, radius, device=self.device)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encoder(x)
        q_z, p_z = self.reparameterize(z_mean, z_var, self.radius)
        z = q_z.rsample()  # sampling using reparameterization trick
        x_rec = self.decoder(z)
        x_cla = self.classifier(z)
        
        return (z_mean, z_var), (q_z, p_z), z, (x_rec, x_cla)
    
