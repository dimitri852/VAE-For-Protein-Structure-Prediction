import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from operator import itemgetter 
from collections import Counter
import matplotlib.pyplot as plt

import torch
from torch.utils.data import  DataLoader 

from data_structure import Protein_Dataset

def load_500(filename):
    """Loads the file.
    output: Angle pairs - np.array 
            Amino acid labels - list
            Secondary structure - list"""
    
    # Amino Acids labels list
    AA_list = []
    # DSSP symbol labels list
    DSSP_list = []
    # Angles list
    list_of_angles = []
    
    for line in filename.readlines():
        if "#" in line:
            continue
        elif "NT" in line:
            continue
        elif "CT" in line:
            continue
        else:
            elemennts_of_line = line.split()
            AA_list.append(elemennts_of_line[0])                 # append Amino Acid (AA) label
            DSSP_list.append(elemennts_of_line[1])               # append Dictionary of Protein Secondary Structure (DSSP) label
            list_of_angles.append(float(elemennts_of_line[2]))   # append phi
            list_of_angles.append(float(elemennts_of_line[3]))   # append psi
    # list --> numpy       
    angle_array = np.array(list_of_angles)       
    # reshape to numpy:[ phi, psi ]
    angle_array = np.reshape(angle_array,((int(len(angle_array)/2)), 2)) 

    return(angle_array, AA_list, DSSP_list)

 
def sensor_or_not(list_of_DSSP, Num_of_classes):
    '''Sensoring of Protein Secondary Structure (DSSP) labels - 2 different methods
        + plotting
        '''
    if  Num_of_classes == 3:       
        # B to E
        list_of_DSSP = [dssp.replace('B', 'E') for dssp in list_of_DSSP]
        # G to E
        list_of_DSSP = [dssp.replace('G', 'H') for dssp in list_of_DSSP]
        # Rest to coil
        list_of_DSSP = [dssp.replace('S', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('T', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('-', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('I', 'C') for dssp in list_of_DSSP]
        
        keys_DSSP = Counter(list_of_DSSP).keys()      
        freqs_DSSP =  Counter(list_of_DSSP).values() 
        
        # Turn to np.array
        list_of_DSSP = np.array(list_of_DSSP)
        list_of_DSSP[list_of_DSSP == 'E'] = 0           # E = 0
        list_of_DSSP[list_of_DSSP == 'H'] = 1           # H = 1
        list_of_DSSP[list_of_DSSP == 'C'] = 2           # C = 2
          
        list_of_DSSP = list_of_DSSP.astype(int)
        y_pos = list(range(0, Num_of_classes))
        # Create bars
        plt.bar(y_pos, freqs_DSSP , width = 0.5, color = 'b')
        # Create names on the x-axis
        plt.xticks(y_pos,  keys_DSSP)
        plt.title("Frequencies of DSSP symbols in dataset")
        #restart Kernel in Error: TypError:'str' object is not callable'
        # Show graphic
        fig = plt.figure()
        
    elif Num_of_classes == 8:
        '''Keep the 8 classes'''
        keys = Counter(list_of_DSSP).keys()      
        freqs =  Counter(list_of_DSSP).values()
        # Turn to np.array
        list_of_DSSP = np.array(list_of_DSSP)
        list_of_DSSP[list_of_DSSP == 'B'] = 0
        list_of_DSSP[list_of_DSSP == 'E'] = 1
        list_of_DSSP[list_of_DSSP == 'G'] = 2
        list_of_DSSP[list_of_DSSP == 'H'] = 3
        list_of_DSSP[list_of_DSSP == 'S'] = 4
        list_of_DSSP[list_of_DSSP == 'T'] = 5
        list_of_DSSP[list_of_DSSP == '-'] = 6
        list_of_DSSP[list_of_DSSP == 'I'] = 7
        
        list_of_DSSP = list_of_DSSP.astype(int)
        y_pos = list(range(0, Num_of_classes))
        # Create bars
        plt.bar(y_pos, freqs, width = 0.5, color = 'b')
        # Create names on the x-axis
        plt.xticks(y_pos,  keys)
        plt.title("Frequencies of DSSP symbols in dataset")
        # Show graphic
        fig = plt.figure()
        print(" 'I' has only 20 occurencies in whole dataset")
    else:
        print('Error: Please select labels:3 or 8')
        
    return(list_of_DSSP, fig)


def AA_to_num(list_of_AA):
    """input: list of  amino acid labels
        output: replaces labels with numbers + plot"""
        
    # Check for imbalance in data labels                         
    # counts the elements' frequency
    keys_AA = Counter(list_of_AA).keys()      
    freqs_AA =  Counter(list_of_AA).values() 
#    print("Check correspondance of freq with spesific AA", list_of_AA.count('C'))
    
    index_list = ['A','R','N','D','C','Q','E','G','H',
                  'I','L','K','M','F','P','S','T','W','Y','V']

    numbers_list = []
    for i in list_of_AA:
        number = index_list.index(i)
        numbers_list.append(number)

    list_of_AA_num = np.array(numbers_list)
    list_of_AA_num = list_of_AA_num.astype(int)
    print(len(list_of_AA_num))
    y_pos = list(range(0,20))
    # Create bars
    plt.bar(y_pos, freqs_AA, color = 'purple')
    # Create names on the x-axis
    plt.xticks(y_pos, keys_AA)
    plt.title("Frequencies of Amino acids in dataset")
    # Show graphic
    fig = plt.figure()   
    return(list_of_AA_num, fig)


def setup_data_loaders( dataset, batchsize, use_cuda = False):
    '''Separate 80/20 the dataset  
    Return: train/TEST loader , test set'''  
    
    torch.manual_seed(0)    # For same random split of train/test set every time the code runs!
    
    # Print shapes of dataset tensors
    print (" ------------------------------------------------ ")
    print ("  --> Size of whole dataset:", len(dataset)," examples")
    print (" ")
    print ("      Angles:", dataset.Angles.shape, dataset.Angles.dtype,dataset.Angles.is_cuda)
    print ("      AA:    ",    dataset.AA.shape, dataset.AA.dtype)
    print ("      DSSP:  ",   dataset.DSSP.shape, dataset.DSSP.dtype)
    print (" ------------------------------------------------ ")

    # SPLIT train/test set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Randomly splits the dataset into non-overlapping new datasets of given lengths
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print (" Size of train set:",  len(train_set)," examples") 
    print (" Size of TEST set: ",   len(test_set)," examples")  
    print (" ------------------------------------------------ ")
    print (" Size of each batch:", batchsize,"examples")  
    
    # CONSTRUCT Dataloaders   
    kwargs = {'num_workers':0, 'pin_memory': use_cuda}        
                                                                    
    train_loader = DataLoader( train_set, 
                        batch_size = batchsize, 
                        shuffle = True,                                               
                        **kwargs)  # Running on CPU
    
    TEST_loader = DataLoader( test_set, 
                        batch_size = batchsize, 
                        shuffle = False,                                               
                        **kwargs)  # Running on CPU
    
    # Check size of train/test batch
    print (' train_loader size: ', len(train_loader),'batches')
    print (' TEST_loader size:  ',  len(TEST_loader),' batches')

    return(train_loader,TEST_loader, test_set )


def process_dataset(sensor, batch_size, use_cuda):
    '''Use previous functions to process the dataset
    input: sensoring method, batch size, CUDA
    output: train_loader, TEST_loader, test_set'''
    
    # Get Everything from the dataset (apart NT CT regions...)
    all_angles, AA, DSSP = load_500(open(r'top500_dataset.txt')) 
    # Sensoring with either of 2 methods: 3 or 8                                                                             
    DSSP, fig = sensor_or_not(DSSP,sensor)
    # Change aminiacid classes with numbers
    AA, fig = AA_to_num(AA)
    # custom Protein_Dataset class
    DATASET = Protein_Dataset(all_angles, AA, DSSP)   
    # GET train/test data in dataloader form and test_set
    train_loader, TEST_loader, test_set = setup_data_loaders( DATASET, batch_size, use_cuda)
    
    return(train_loader, TEST_loader, test_set)    
    
    
def extract_elements_of_test_set(test_set):
    '''input: test set
        output: Features (in tensor form) of test set: Angles, Amino labels, Secondary labels '''
    import numpy as np
    
    test_angles = []
    test_acids = []
    test_dssp  = []
    
    for x,AA,DSSP in test_set:
        # convert to np and detach in case of GPU 
        angles = x.cpu().detach().numpy()
        # append angles to list
        test_angles.append(angles)
        
        labels = AA.cpu().detach().numpy()
        #append labels
        test_acids.append(labels)
        
        secondary_s = DSSP.cpu().detach().numpy()
        # append secondary structure label
        test_dssp.append(secondary_s)
        
    # convert lists to numpy
    test_angles = np.array(test_angles)
    test_acids = np.array(test_acids)
    test_dssp = np.array(test_dssp)
    # convert again to torch
    x = torch.from_numpy(test_angles)
    AA = torch.from_numpy(test_acids)
    DSSP = torch.from_numpy(test_dssp)
    
    return x, AA, DSSP


def Ramachandran_plot( data_angles,plot_title ):
    '''Makes Ramachandran plot '''
    import numpy as np
    from matplotlib.colors import LogNorm
    #convert radians to degrees
    Degrees = np.rad2deg(data_angles) 
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    plt.figure(figsize=(7, 6))
    plt.hist2d( phi, psi, bins = 200, norm = LogNorm(), cmap = plt.cm.jet )
    plt.title(plot_title)

    plt.xlabel('φ')
    plt.ylabel('ψ')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.colorbar()
    fig = plt.figure()
    return(fig)    

 
def Rama2(dataset):
    '''Makes Ramachandran plot using kde
        - super slow'''
    from scipy.stats import kde
    #convert radians to degrees
    Degrees = np.rad2deg(dataset) 
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    
    nbins=100
    k = kde.gaussian_kde([phi,psi])
    xi, yi = np.mgrid[phi.min():phi.max():nbins*1j, psi.min():psi.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
 
    plt.title('Rama')
    plt.xlabel('phi')
    plt.ylabel('psi')

    fig = plt.figure()
    return(fig)
   

def Specific_Amino_Rama(dataset, AminoAcids, title):
    '''Makes Ramachandran plots based on the 
    specific amino acids-angles taken from dataset:
    A , R , N , D , C , Q , E , G , H , I , L  , K  , M  , F  , P  , S  , T  , W  , Y  , V
    0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8, 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19
    input:  - dataset created from MyDataset Class
            - list of spesific amino acids
            - title of graph
        '''
    amino_acid_list = []
    # iterate over amino acids in input: list of spesific amino acids
    for AA in AminoAcids:
        # Get index of spesific amino acid in dataset 
        for i,x in enumerate(dataset.AA):
            if x == AA:
                amino_acid_list.append(i)
               
    # Get phi,psi angles based on the index list above
    angles = list(itemgetter(*amino_acid_list)(dataset.Angles.numpy()))
    # Convert to numpy
    angles = np.array(angles)
    # Convert radians to degrees
    degrees = np.rad2deg(angles) 
    
    # Get phi, psi angles from dataset
    phi = degrees[:,0] 
    psi = degrees[:,1] 
    plt.scatter(phi, psi, s=1)
    plt.title( title)
    plt.xlabel('phi')
    plt.ylabel('psi')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    fig=plt.figure()
    return(fig)
    

def plot_ELBOs(train_elbo, test_elbo,test_frequency):
    """ Plots Training and Test error"""
    plt.figure(figsize = (8, 8))
    test_elbo = np.array(test_elbo)
    train_elbo = np.array(train_elbo)
    list_of_epochs=[]
    
    for i in range(0,len(train_elbo), test_frequency):
        list_of_epochs.append(i)
    list_of_epochs = np.array(list_of_epochs)
    
    # for test ELBO
    data = np.concatenate([list_of_epochs[:, sp.newaxis], test_elbo[:, sp.newaxis]], axis=1)
    
    # for training ELBO
    data2 = np.concatenate([np.arange(len(train_elbo))[:, sp.newaxis], train_elbo[:, sp.newaxis]], axis=1)
    
    # Plot every N training epochs the test ELBO->Test Frequency
    df = pd.DataFrame(data = data, columns=['Training Epoch', 'Test ELBO'])
    df2 = pd.DataFrame(data = data2, columns=['Training Epoch', 'Train ELBO'])

    plt.plot( 'Training Epoch', 'Test ELBO', data = df, marker = 'o', markerfacecolor = 'blue', markersize = 1, color = 'blue', linewidth = 1)
    plt.plot( 'Training Epoch', 'Train ELBO', data = df2, marker = 'o', markerfacecolor = 'red', markersize = 1, color = 'red', linewidth = 1)
    plt.title('Train / Test Loss', fontsize = 14, loc = 'center')
    plt.yscale('log')    
    plt.ylim((100, 1200))   
    plt.show()
    sns.set_style("ticks")


