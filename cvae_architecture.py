import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import InverseAutoregressiveFlow, TransformedDistribution
from pyro.nn import AutoRegressiveNN
import von_mises

import torch.nn as nn
import torch
import math

from data_utils import extract_elements_of_test_set


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, amino_acid_labels):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear((2 + amino_acid_labels), hidden_dim)  # 2 + 20
        self.fc2 = nn.Linear( hidden_dim, hidden_dim) 
        self.fc31 = nn.Linear(hidden_dim, z_dim)   
        self.fc32 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()
        
    def forward(self, x):
        """Forward pass of the Endoder architecture
            Input : dihedral angles
            output: latent space"""
        hidden = self.softplus1(self.fc1(x)) 
        hidden = self.softplus2(self.fc2(hidden))  

        # mean vector, (positive) square root covariance
        # size: batch_size x z_dim
        z_loc = self.fc31(hidden)
        # We add a very small positive quantity
        # to avoid exploding gradients
        z_scale = 0.00001 + torch.exp(self.fc32(hidden)) 
        return z_loc, z_scale
        
    
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, amino_acid_labels, positive_constant):
        super(Decoder,self).__init__()
        self.fc1 = nn.Linear((z_dim + amino_acid_labels), hidden_dim)        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    
        self.fc31 = nn.Linear(hidden_dim, 2)   # m  for phi,psi: 2 dimensions
        self.fc32 = nn.Linear(hidden_dim, 2)   # k  for phi,psi: 2 dimensions
        self.positive_constant = positive_constant
        
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward pass of the Endoder architecture
            Input : latent space
            output: von Mises distribution params"""
        
        # Define forward computation of z
        hidden = self.softplus1(self.fc1(z))
        hidden = self.softplus2(self.fc2(hidden))
        
        # Return the parameters of von misses distribution: mi , kappa
        loc_img = (self.sigmoid(self.fc31(hidden))) * 2 * math.pi  - math.pi             
        k = self.positive_constant  + self.relu(self.fc32(hidden))
        
        return loc_img, k
     

class CVAE(nn.Module):
    def __init__(self, z_dim, hidden_dim, amino_acid_labels,
                             positive_constant, num_iafs, iaf_dim, use_cuda):
        super(CVAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim,amino_acid_labels)
        self.decoder = Decoder(z_dim, hidden_dim,amino_acid_labels, positive_constant)
        self.z_dim = z_dim
        
        # In case we want to try normalizing flows
        self.iafs = [InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim])) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.use_cuda = use_cuda
        
        # Enable Cuda or not
        if use_cuda:
            # calling cuda() here will put all the parameters of        
            # the encoder and decoder networks into gpu memory
            self.cuda()
   
    def guide(self, x, AA, annealing_factor = 1.0):
        
        """ Defines the guide (i.e. variational distribution) q(z|x) """
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):  # x.shape[0] defines the independance among examples in the batch
            # Concatenate torsion angles 'x' and amino acid label 'AA'
            xAA = torch.cat((x, AA), 1)
            # Use the encoder to get the parameters of q(z|x,y)  
            z_loc, z_scale = self.encoder.forward(xAA)
            
            with pyro.poutine.scale(scale = annealing_factor):   
                # If we are using normalizing flows, we apply the sequence 
                # of transformations parameterized by self.iafs
                if self.iafs.__len__() > 0:
                    # Output of normalizing flowÆ all dimensions are correlated (event shape is not empty)                    
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)

                    # -- KL annealing - Enclose latent sample statement with a poutine context, both guide & model
                    pyro.sample("latent", z_dist.to_event(1))

                else:
                    # when NO normalizing flows are used,
                    # sample the latent code z without the transformations
                    # -- KL annealing - Enclose latent sample statement with a poutine context, both guide & model
                    pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
                
    def model(self, x, AA, annealing_factor = 1.0):
        """Defines the model p(x|z)p(z)"""
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
        
            # In case we’re on GPU, we use new_zeros and new_ones to ensure
            # that newly created tensors are on the same GPU device.
            
            # Setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            
            # Sample from prior (values will be sampled by guide when computing the ELBO)
            with poutine.scale(None, annealing_factor):
                # -- KL annealing - Enclose latent sample statement with a poutine context, both guide & model
                z = pyro.sample("latent", dist.Normal(z_loc,z_scale).to_event(1))
            # Concatenate latent code z and amino acid label 'AA'
            zAA = torch.cat((z,AA), 1)
            # Decode the latent code + Label
            loc_img, k = self.decoder.forward(zAA)
            # score against actual angles
            pyro.sample("obs", dist.von_mises.VonMises(loc_img, k).to_event(1), obs = x)
    
    def reconstruct_test_set(self, test_set):
        ''' Sampling from the Data...
        if we learned good values for θ and φ (params),
        the real Ramachandran plot of test set and 
        the reconstructed Ramachandran plot of the test set 
        should be similar.(VAE Pyro tutorial)'''
        
        x, AA, DSSP = extract_elements_of_test_set(test_set)
          
        # Load TEST data to the already trained neural networks
        # Concatenate torsion angles and amino acid labels
        xAA = torch.cat((x,AA), 1)
        # ------> ENCODE
        z_loc, z_scale = self.encoder(xAA)
        z = dist.Normal(z_loc,z_scale).sample()
        # Concatenate Zs and amino acid labels
        zAA = torch.cat((z,AA), 1)
        # ------> DECODE (note we do NOT sample in angle space)
        means, kappa = self.decoder(zAA)
        
        # output a numpy array to make the Ramachandran plots
        means = means.detach().numpy()  
        return means

    def generate(self, number_of_samples, amino_acid):
        ''' Generative process:
        Samples random noise from prior P(z) and
        concatenates it with the specific secondary stucture label,
        in order to pass it through the Decoder.
        
        Input:  1) number_of_samples
                2) amino_acid label

       output:  1) means
                2) ks,
                3) generated angles
        '''
        # Generate one hot label 'y' of specific Amino Acid 
        # in order to concatenate them later with z
        labels = torch.eye(20,20)
        
        # Grab the right amino acid 1hot label
        if amino_acid == amino_acid: 
            AA = labels[amino_acid,:].reshape(1,20)
        else:
            raise ValueError("""The provided Amino acid parameter has invalid values,
                                please select amino acids:
                                A-0, R-1, N-2, D-3, C-4, Q-5, E-6, G-7, H-8, I-9, L-10,
                                K-11, M-12, F-13, P-14, S-15, T-16, W-17, Y-18, V-19""")
        list_of_mean_angles = []
        list_of_k = []
        list_of_generation = []
        
        for i in range(number_of_samples):      
            # Sample from P(z) N(0,1) tensors of size[1,z_dim]
            z_loc = torch.zeros(1, self.z_dim)
            z_scale = torch.ones(1,self.z_dim)
            z = dist.Normal(z_loc, z_scale).sample()
            
            # decode the samples concatenated with the onehot label 'y'
            # (note we don't sample in angles space)       
            zAA = torch.cat((z, AA), 1)            
            loc_img, k = self.decoder(zAA)
            
            # generate angles after sampling  from von Mises
            angles_generated = von_mises.VonMises(loc_img,k).sample()
            
            # append the mean angles and ks
            list_of_mean_angles.append(loc_img)
            list_of_k.append(k)
            # append the generated angles after sampling  from von Mises
            list_of_generation.append(angles_generated)

            # Sanity Check: shapes after last sample is created
            if i == (number_of_samples-1):
                print("########################################################")
                print("Sanity Checks:")
                print(loc_img.shape,           "   :: mean_phi , mean_psi : 2 dim")
                print(k.shape,                 "   :: k_phi    , k_psi    : 2 dim")
                print(angles_generated.shape,  "   :: gen_phi  , gen_psi  : 2 dim")
            else:
                pass
        
        stacked_means = torch.stack(list_of_mean_angles)
        stacked_means = stacked_means.view(number_of_samples, 2)
        
        print(stacked_means.shape,' :: means')
        stacked_k = torch.stack(list_of_k)
        stacked_k = stacked_k.view(number_of_samples, 2)
        
        print(stacked_k.shape,' :: k')
        stack_gen = torch.stack(list_of_generation)
        stack_gen = stack_gen.view(number_of_samples, 2)
        print(stack_gen.shape,' :: generated angles')
        
        print('########################################################')    
        stacked_means = stacked_means.cpu().detach().numpy() 
        stacked_k = stacked_k.cpu().detach().numpy() 
        stack_gen = stack_gen.cpu().detach().numpy() 
        return stacked_means, stacked_k, stack_gen            
            
    
