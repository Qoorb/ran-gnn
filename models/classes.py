from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
import pandas as pd
import torch
import numpy as np
from torch_geometric.nn import knn_graph


class PointCloudDataset(Dataset):
    def __init__(self, csv_files,num_points,k):
        self.csv_files = csv_files
        self.num_points = num_points
        self.k = k

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        num_points = self.num_points
        k = self.k
        data = pd.read_csv(csv_file)
        data = data.iloc[:num_points]

        points = data[['X', 'Y', 'Z']].values.astype('float32')
        next_points = data[['X1', 'Y1', 'Z1']].values.astype('float32')
        points = self.normalize(points)
        next_points = self.normalize(next_points)

        edge_index = knn_graph(torch.tensor(points), k=k, loop=False)

        return Data(x=torch.tensor(points), edge_index=edge_index, y=torch.tensor(next_points).unsqueeze(0))  # Add batch dimension

    def normalize(self, data):
        center = np.mean(data, axis=0)
        radius = np.max(np.linalg.norm(data - center, axis=1))
        data_centered = data - center
        normalized_data = data_centered / radius
        return normalized_data