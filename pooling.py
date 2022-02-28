import torch
import torch.nn as nn

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
