import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DGCNN(nn.Module):
    def __init__(self, k=3, emb_dims=1024, output_channels=3):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.linear1 = nn.Linear(64*3 + 128, emb_dims)
        self.bn1 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(emb_dims, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout()
        self.linear3 = nn.Linear(512, output_channels)

    def get_graph_feature(self, x, k=20):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        
        idx = self.knn(x, k=k)   # (batch_size, num_points, k)
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        _, num_dims, _ = x.size()
        
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        
        return feature

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1).contiguous()

        x1 = self.get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1).max(dim=-1, keepdim=False)[0]
        
        x2 = self.get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2).max(dim=-1, keepdim=False)[0]
        
        x3 = self.get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3).max(dim=-1, keepdim=False)[0]
        
        x4 = self.get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4).max(dim=-1, keepdim=False)[0]
        
        x1 = torch.max(x1, 2)[0]
        x2 = torch.max(x2, 2)[0]
        x3 = torch.max(x3, 2)[0]
        x4 = torch.max(x4, 2)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x


