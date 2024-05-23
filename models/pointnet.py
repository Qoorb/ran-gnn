import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, DataLoader

class PointNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PointNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x