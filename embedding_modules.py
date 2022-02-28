import torch
import torch.nn as nn

class Naive_Embedding(nn.Module):
    def __init__(self, input_features, output_features):
        super(Naive_Embedding, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.embedding = nn.Sequential(nn.Conv1d(self.input_features, self.output_features, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(self.output_features),
                                       nn.ReLU(),
                                       nn.Conv1d(self.output_features, self.output_features, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(self.output_features),
                                       nn.BatchNorm1d(self.output_features))
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.embedding(x)
        return x
