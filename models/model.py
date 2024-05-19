import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        out = torch.cat([x[row], x[col] - x[row]], dim=-1)
        return self.mlp(out)

class DGCNN(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = EdgeConv(in_channels, 64)
        self.conv2 = EdgeConv(64, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def knn_graph(self, x, k):
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(x.cpu().numpy())
        knn_indices = knn.kneighbors(x.cpu().numpy(), return_distance=False)
        row = torch.arange(x.size(0)).repeat_interleave(k)
        col = torch.from_numpy(knn_indices.flatten())
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, x):
        edge_index = self.knn_graph(x, self.k).to(x.device)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)
        return x


