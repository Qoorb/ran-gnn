from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np


class PointCloudDataset(Dataset):
    def __init__(self, csv_files):
        self.csv_files = csv_files

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        data = pd.read_csv(csv_file)

        points = data[['X', 'Y', 'Z']].values.astype('float32')
        next_points = data[['X1', 'Y1', 'Z1']].values.astype('float32')

        # Нормализация данных
        points = self.normalize(points)
        next_points = self.normalize(next_points)

        # Транспонирование для соответствия форме (num_dims, num_points)
        points = points.T
        next_points = next_points.T

        return torch.tensor(points), torch.tensor(next_points)
    
    def normalize(self, data):
        center = np.mean(data, axis=0)
        radius = np.max(np.linalg.norm(data - center, axis=1))
        data_centered = data - center
        normalized_data = data_centered / radius
        return normalized_data
