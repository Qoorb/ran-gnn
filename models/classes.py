from torch.utils.data import Dataset
import pandas as pd
import torch

class PointCloudDataset(Dataset):

    '''This class needed for creating dataloader.
    Class'es input take point place list 
    Points - our point object, Next_point - point object in next state(target)'''

    def __init__(self, csv_files):
        self.csv_files = csv_files

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        data = pd.read_csv(csv_file)

        points = data[['X','Y','Z' ]].values.astype('float32')
        next_points = data[['X1','Y1','Z1' ]].values.astype('float32')
        
        return torch.tensor(points), torch.tensor(next_points)
