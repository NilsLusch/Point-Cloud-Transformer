import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points,knn_gather, sample_farthest_points

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
#Based on the code form: https://github.com/MenghaoGuo/PCT/blob/main/networks/cls/pct.py
class Sample_and_Group(nn.Module):
    def __init__(self, output_features, k):
        super(Sample_and_Group, self).__init__()
        self.output_features = output_features
        self.k = int(k)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.output_features,self.output_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.output_features),
            nn.ReLU(),
            nn.Conv1d(self.output_features, self.output_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.output_features),
            nn.ReLU(),
        )
    #coordinates are the unembedded points, distance based operations on the point cloud are done based on the distances
    #within the original unembedded point cloud
    def forward(self, x, coordinates, sample=None):
        #3D_ops require the format (Batch, points, point_features) 
        batch, points, dimensions = x.size()
        if sample != None:
            points = sample
            points = int(points)
            #Farthest point sampling for the coordinates
            sample_coords, sample_idx = sample_farthest_points(coordinates, K=int(sample))
            #Get the indicies of the nearest neighboors
            _,knn_idx,_ = knn_points(sample_coords, coordinates, K=self.k)
            #find the nearest neighboors in the embedded point cloud based on the indices
            neighboors  = knn_gather(x, knn_idx)
            #Apply the farthest point sampling to x
            x = x[torch.arange(x.shape[0]).unsqueeze(-1), sample_idx]
        else:
            sample_coords = coordinates
            #Get the indicies of the nearest neighboors
            _,knn_idx,_ = knn_points(sample_coords, coordinates, K=self.k)
            #Find the nearest neighboors based on the indices
            neighboors  = knn_gather(x, knn_idx)
        #Subtract the orginal points from its k-nearest neighboors
        x = x.view(batch, points, 1 , dimensions)
        sub_neighboors = neighboors - x
        #concatinate x with the distances to its k-nearest neighboors
        x = torch.cat((sub_neighboors, x.repeat(1,1,self.k,1)), dim=-1)
        #Treat every point as it's own batch to make the operation work with 1d convolutions
        x = x.permute(0, 1, 3, 2).reshape(batch*points, dimensions*2, self.k)
        x = self.conv_layers(x)
        #Remove the neighboor dimension by applying max pooling
        x,_ = torch.max(x,2)
        #seperate batch and points again
        x = x.reshape(batch, points, -1)
        #Output will be (batch, dimensions, points) and the sampled coordiantes
        return x, sample_coords
class Neighboorhood_Embedding(nn.Module):
    #input features refers to the amount of features in the set for example xyz or additional rgb features
    #output features referers to the features output after the two cascading layers
    #samplingdetermines if samplingis used only used in classification
    #num_samples if sampling is used list of number of points to sample to
    def __init__(self, input_features, output_features, k, sampling, positional_embedding=False):
        super(Neighboorhood_Embedding, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.k = k
        self.sampling = sampling
        self.positional_embedding = positional_embedding
        
        self.input_embedding = nn.Sequential(nn.Conv1d(self.input_features, self.output_features//4, kernel_size=1, bias=False),
                      nn.BatchNorm1d(self.output_features//4),
                      nn.ReLU(),
                      nn.Conv1d(self.output_features//4, self.output_features//4, kernel_size=1, bias=False),
                      nn.BatchNorm1d(self.output_features//4),
                      nn.ReLU())
        
        self.Sample_and_Group_1 = Sample_and_Group(output_features=self.output_features//2, k=self.k)
        self.Sample_and_Group_2 = Sample_and_Group(output_features=self.output_features, k=self.k)
        
        if self.positional_embedding == True:
            self.pos_embedding = nn.Conv1d(self.input_features, self.output_features, kernel_size=1, bias=False)
        
        self.out_layers = nn.Sequential(nn.Conv1d(output_features, output_features, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(output_features))

    def forward(self, x):
        coordinates = x
        x = x.permute(0,2,1)
        x = self.input_embedding(x)
        x = x.permute(0,2,1)
        
        if self.sampling != None:
            assert 0 < self.sampling < 1, "sampling size must be between 0 and 1"
            sample =  x.size(1)*self.sampling
            x, coordinates = self.Sample_and_Group_1(x, coordinates, sample*2)
            x, coordinates = self.Sample_and_Group_2(x, coordinates, sample)
        else:
            x,_ = self.Sample_and_Group_1(x, coordinates)
            x,_ = self.Sample_and_Group_2(x, coordinates)
        
        #Return x to the correct shape for the conv-blocks
        x = x.permute(0,2,1)
        x = self.out_layers(x)
        #additional positional embedding implemented in the Git repo of the paper author 
        #https://github.com/MenghaoGuo/PCT/blob/main/networks/cls/pct.py
        if self.positional_embedding == True:
            coordinates = coordinates.permute(0,2,1)
            coordinates = self.pos_embedding(coordinates)
        return x, coordinates
