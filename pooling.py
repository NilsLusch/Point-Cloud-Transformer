import torch
import torch.nn as nn
from pytorch3d.ops import knn_points,knn_gather, sample_farthest_points

class MA_Pooling(nn.Module):
    def __init__(self, pooling):
        super(MA_Pooling, self).__init__()
        self.pooling = pooling
        
    def forward(self, x):
        if self.pooling in {"both","max"}:
            max_pool,_ = torch.max(x,2)
            if self.pooling == "max":
                    x = max_pool
        if self.pooling in {"both","avg"}:
            avg_pool = torch.mean(x,2)
            if self.pooling == "avg":
                    x = avg_pool
        if self.pooling == "both":
            x = torch.cat((max_pool, avg_pool), dim=1)
        return x

class KNN_Pooling(nn.Module):
    def __init__(self, sample, k):
        super(KNN_Pooling, self).__init__()
        self.sample = int(sample) 
        self.k = int(k)
    def forward(self, x, coordinates):
        x = x.permute(0,2,1)
        sample_coords, sample_idx = sample_farthest_points(coordinates, K=self.sample)
        _,knn_idx,_ = knn_points(sample_coords, coordinates, K=self.k)
        x  = knn_gather(x, knn_idx)
        x = torch.max(x,2)[0]
        x = torch.flatten(x, start_dim=1)
        return x
