

def get_minibatch_annealing_factor(epoch, batch_number, annealing_epochs,
                                   minimum_annealing_factor, train_loader, x, AA):
    ''' KL annealing - Calculates annealing factor for every batch in every epoch
    
    input:  1) epoch right now
            2) index of mini batch
            3) annealing epochs
            4) minimum annealing factor
            5) train loader
            6) torsion angles,
            7) Amino acid label 
            
    output: annealing_factor''' 
    
    # If we use KL annealing        
    if annealing_epochs > 0 and epoch < annealing_epochs:
        # Compute the APPROPRIATE KL annealing factor for the current mini-batch in the current epoch.
        # Kl annealing factor starts from: minimum_annealing_factor and rises linearly up to 1 in final annealing epoch.
        # Both 'minimum_annealing_factor' and 'annealing_epochs' are hyperparameters and set in the beginning of the program.
        annealing_factor = minimum_annealing_factor + (1.0 - minimum_annealing_factor) * \
                            (float(batch_number + epoch * len(train_loader) + 1) /
                             float(annealing_epochs * len(train_loader)))
    # if NOT 
    else:
        # by default the KL annealing factor is unity
        annealing_factor = 1.0
    
    return annealing_factor 


def train(svi, train_loader, epoch,
          annealing_epochs, minimum_annealing_factor,
          use_cuda):
    """Trains the model"""
    # Set an accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
    train_loss = 0.0    
    #,_ ignore DSSP / iterate only over Angles,Amino acids of dataloader + keep batch number index to calculate the annealing factor
    for batch_number,(x, AA, _) in enumerate(train_loader): 

        # CUDA
        # if train on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            AA = AA.cuda()
        
        # Calculate annealing factor of each batch
        annealing_factor = get_minibatch_annealing_factor(epoch, batch_number, annealing_epochs,
                                                          minimum_annealing_factor, train_loader, x, AA )
        # do an ELBO gradient step and accumulate loss
        train_loss +=  svi.step(x, AA, annealing_factor)  
    
    # Normalize loss
    normalizer_train = len(train_loader)
    total_epoch_loss_train = train_loss / normalizer_train  
    return total_epoch_loss_train  


def evaluate(svi, test_loader, use_cuda):
    """Tests the model"""
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    #,_ ignore DSSP / iterate only over Angles,Amino acids of dataloader
    for x,AA,_ in test_loader:   
        
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            AA = AA.cuda()
        
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, AA)   
        
    # Normalize loss                                                        
    normalizer_test = len(test_loader)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test
