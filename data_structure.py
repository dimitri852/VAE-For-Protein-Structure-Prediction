import torch

class Protein_Dataset():
    """ Protein dataset - Enables the use of
    Pytorch Dataloader functionalities"""
    
    def __init__(self, Angles, AA, DSSP):
        
        self.Angles = torch.from_numpy(Angles).float()            
        self.AA = torch.from_numpy(AA).long()
         # create one hot labels for Amino Acid labels
        self.AA = torch.zeros(len(self.AA), (self.AA).max()+1).scatter_(1, (self.AA).unsqueeze(1), 1.)
        self.DSSP = torch.from_numpy(DSSP).long()
        # create one hot labels DSSP labels
        self.DSSP = torch.zeros(len(self.DSSP), (self.DSSP).max()+1).scatter_(1, (self.DSSP).unsqueeze(1), 1.)

    def __len__(self):
        
        return len(self.Angles)    
      
    def __getitem__(self, index):
        
        angle_pairs = self.Angles[index]
        label_1 = self.AA[index]
        label_2 = self.DSSP[index]

        return angle_pairs, label_1, label_2
    
