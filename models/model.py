import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels = 128, num_neighbors=20):
        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(3 * 2, hidden_channels), torch.nn.ReLU()),
                                     k=num_neighbors, aggr='max')
        self.conv2 = DynamicEdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(hidden_channels * 2, hidden_channels), torch.nn.ReLU()),
                                     k=num_neighbors, aggr='max')
        self.conv3 = DynamicEdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(hidden_channels * 2, hidden_channels), torch.nn.ReLU()),
                                     k=num_neighbors, aggr='max')
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 3)  # Выводим XYZ координаты следующего положения точек

    def forward(self, data):
        x, batch = data[0], data.batch[1]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = global_mean_pool(x, batch)  # Глобальное усреднение по всем точкам

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
