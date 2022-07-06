import time
import numpy as np
import os
import matplotlib.pyplot as plt 

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam 

from data_utils import process_dataset, extract_elements_of_test_set, plot_ELBOs, Ramachandran_plot
from cvae_architecture import CVAE
from train_utils import train, evaluate
 
# Checks
assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)
#pyro.set_rng_seed(0)


def save_checkpoint(save_directory, save_name):  
    '''Saves the weights of the model and the optimizer state to the disk'''
    try:
        # Create a folder to store the experiment
        # directory \ folder:save_name
        if not os.path.exists("%s\%s" % ( save_directory, save_name)):
            os.makedirs("%s\%s" % (save_directory, save_name))
            print("Creating directory...")
            
    except OSError:
        print("ERROR: Creating directory:" + save_name)
        
    print("Saving model...") 
    torch.save(cvae.state_dict(), ("%s\%s\MODEL_%s" % (save_directory, save_name, save_name)))

    # Save Optimizer      
    print("Saving optimizer state...")
    optimizer.save("%s\%s\OPTIMIZER_%s" % (save_directory, save_name, save_name))

    # Save Train/Test Loss
    print("Saving train/test loss...")
    np.savetxt("%s\%s\TRAIN_elbo_%s.txt" % (save_directory, save_name,save_name), np.array(train_elbo))
    np.savetxt("%s\%s\TEST_elbo_%s.txt" % (save_directory, save_name,save_name), np.array(test_elbo))
    
    print("Done Saving!")


def load_checkpoint(save_directory, load_name):
    """Loads the weights of the model and the optimizer state from disk"""

    print("Loading Model...")
    cvae.load_state_dict(torch.load("%s\%s\MODEL_%s" % (save_directory, load_name, load_name)))
                                                        
    print("Loading Optimizer state...")
    optimizer.load("%s\%s\OPTIMIZER_%s" % (save_directory, load_name, load_name))
    
    print("Loading train/test loss...")
    train_elbo = list(np.genfromtxt("%s\%s\TRAIN_elbo_%s.txt" % (save_directory, load_name, load_name)))
    test_elbo = list(np.genfromtxt("%s\%s\TEST_elbo_%s.txt" % (save_directory, load_name, load_name)))
    print("Done loading!")
    return (train_elbo, test_elbo)


batch_size = 128
learning_rate = 1.0e-3
NUM_particles = 1 # by default: 1

# define the use of CUDA
use_cuda = False

# Define the dimensions of the building blocks
z_dim = 50
hidden_dim = 100
amino_acid_labels = 20

# Constant value that is added to the relu output and treated also as a hyperparameter
positive_constant = 70       

# Define the total number of epochs for training 
num_epochs = 4
# Check Test loss every # epochs
test_loss_epoch = 2

# KL rises linearly from 'minimum_annealing_factor' to 1,
# batch after batch until it reaches the epoch:'annealing_epochs'.
# If annealing_epochs is set to 0 --> No KL annealing
annealing_epochs = 0
# If min factor is set to 1 --> No KL annealing
minimum_annealing_factor = 0

# Normalizing flows
iaf_dim = 0   # hidden iaf units
num_iafs = 0  # number of normalizing flows          

# Process dataset
train_loader, TEST_loader, test_set = process_dataset(3, batch_size, use_cuda)
   
# Ramachandran plot for test_set
test_x, test_AA, test_Dssp = extract_elements_of_test_set(test_set)
test_x = test_x.numpy()
plt.show(Ramachandran_plot(test_x, 'Ramachandran plot (test set)'))

# Save every # epochs
check_point_epoch = 2
save_directory = (r"your dir")
# Saves file name id - Important: The name changes in accordance with the hyperparameters
# choosen per code execution, creates a unique and ditinquishable file name
save_name = "CVAE_h%d_hu%d_zd%d_ReL%d_batch%d_iaf%d_iafDim%d_anealep%d_minAnFac%s_Par%d_Adam" % (2 , hidden_dim, z_dim, positive_constant, batch_size, num_iafs,
                                                                                                 iaf_dim, annealing_epochs, minimum_annealing_factor, NUM_particles)
# Clear param store or load from checkpoint
pyro.clear_param_store()
# example of reloaded file
#train_elbo, test_elbo = load_checkpoint(save_directory,'5.12CVAE_h2_hu100_zd50_ReL70_batch128_iaf0_iafDim50_anealep800_minAnFac0.1_Par1_Adam124')

# 1 - Setup the Variational Autoencoder Model
# If you need to move a model to GPU via .cuda(), please do so before constructing the 
# optimizers for it (https://pytorch.org/docs/stable/optim.html).
cvae = CVAE(z_dim = z_dim, hidden_dim = hidden_dim, amino_acid_labels = amino_acid_labels,
                             positive_constant = positive_constant , num_iafs = num_iafs, iaf_dim = iaf_dim, use_cuda = use_cuda)

# 2 - Setup the Optimizer - Adam
adam_args = {"lr": learning_rate}
optimizer = Adam(adam_args)

# 3 - Setup the inference algorithm
svi = SVI(cvae.model, cvae.guide, optimizer, loss = Trace_ELBO(NUM_particles))

    #################
    # TRAINING LOOP #
    #################
    
train_elbo = [] # Loss (-ELBO)
test_elbo = []

start = time.time()  

# In case we continue training from a check point, len(train_elbo) serves as new starting epoch.
for epoch in range(len(train_elbo),(num_epochs)): 
    
    if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0 :
        plot_ELBOs((train_elbo), (test_elbo), test_loss_epoch)
        # Saving lives
        save_checkpoint(save_directory, save_name)
        
    total_epoch_loss_train = train(svi, train_loader, epoch,
                                   annealing_epochs, minimum_annealing_factor,
                                   use_cuda )
    # Report training diagnostics
    train_elbo.append(total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
    
    # Every # epochs ---> test
    if epoch % test_loss_epoch == 0:
        # Report test diagnostics
        total_epoch_loss_test = evaluate(svi, TEST_loader, use_cuda)
        test_elbo.append(total_epoch_loss_test)
        print("[epoch %03d] average TEST loss: %.4f" % (epoch, total_epoch_loss_test))
        
    # When reaches the final training epoch    
    # --plot Train/Test ELBO
    if epoch == (num_epochs-1):
        plot_ELBOs((train_elbo), (test_elbo), test_loss_epoch)
     
end = time.time()
print('Time waiting patiently as the Error goes down:', end - start)

# Generative Process - Plot amino acids 
number_of_samples = 16000
amino_acid_label = 7 # different numbers from 0-19 plot different amino acid
gener_means, ks, gener_angles = cvae.generate(number_of_samples, amino_acid_label)
plt.show(Ramachandran_plot(gener_angles, 'Generated sampled angles'))
plt.show(Ramachandran_plot(gener_means, 'Generated mean angles' ))  

# Check weights of NNs through recunstruction of test set
rec_means = cvae.reconstruct_test_set(test_set)
plt.show(Ramachandran_plot(rec_means, 'Ramachandran plot of Reconstructed test set')) 

