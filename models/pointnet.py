import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.nn import DenseGCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


class PointNet(torch.nn.Module):

    def __init__(self,):
        super(PointNet, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_points, num_classes):
        super(Net, self).__init__()
        self.conv1 = DenseGCNConv(num_features, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)

        self.num_points = num_points

    def forward(self, x, edge_index, batch):
        adj = to_dense_adj(edge_index, batch=batch)
        x, mask = to_dense_batch(x, batch=batch)
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x



class DGCNN(torch.nn.Module):
    def __init__(self, input_dim, num_points, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_points = num_points

        self.conv1 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * input_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        ), k=k)

        self.conv2 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * 64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        ), k=k)

        self.conv3 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * 128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        ), k=k)

        self.conv4 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(2 * 256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        ), k=k)

        self.lin1 = torch.nn.Linear(512, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.lin3 = torch.nn.Linear(128, num_points * 3)
        self.num_points = num_points
        self.output_dim = 3

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        x = global_mean_pool(x4, batch)

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))

        x = self.lin3(x)
        x = x.view(-1, self.num_points, self.output_dim)
        return x