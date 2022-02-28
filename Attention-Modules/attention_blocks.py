import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention import Self_Attention_Block, Multihead_Attention_Layer, Decoder_Block, Decoder_Block_2

class Self_Attention_Block(nn.Module):
    def __init__(self, layers, attention_features, key_size, value_size):
        super(Self_Attention_Block, self).__init__()
        self.layers = layers
        self.attention_features = attention_features
        self.key_size = key_size
        self.value_size = value_size
        
        self.attention_layers =  nn.ModuleList([Offset_Attention_Layer(self.attention_features, self.key_size, self.value_size) 
                                                                        for i in range(self.layers)])
    def forward(self, x, coordinates=None):
        output = []
        for count, layer in enumerate(self.attention_layers):
            if count == 0:
                 out = layer(x, coordinates)
            else:
                out = layer(output[count-1], coordinates)
            output.append(out)   
            
        x = torch.cat(output, dim=1)
        return x
      
class MultiHead_Attention_Block(nn.Module):
    def __init__(self, layers, heads, embeddsize, forward_expansion, dropout):
        super(MultiHead_Attention_Block, self).__init__()
        self.layers = layers
        self.heads = heads
        self.embeddsize = embeddsize
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        
        self.decoder_layers =  nn.ModuleList([Decoder_Block(self.embeddsize, self.heads, self.forward_expansion, self.dropout) 
                                                                        for i in range(self.layers)])
    def forward(self, x, coordinates=None):
        output = []
        for count, layer in enumerate(self.decoder_layers):
            if count == 0:
                 out = layer(x, coordinates)
            else:
                out = layer(output[count-1], coordinates)
            output.append(out)   
        
        x = torch.cat(output, dim=1)
        return x

#This only outputs the last results of the last layer       
class MultiHead_Attention_Block2(nn.Module):
    def __init__(self, layers, heads, embeddsize, forward_expansion,dropout):
        super(MultiHead_Attention_Block2, self).__init__()
        self.layers = layers
        self.heads = heads
        self.embeddsize = embeddsize
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        
        self.decoder_layers =  nn.ModuleList([Decoder_Block(self.embeddsize, self.heads, self.forward_expansion, self.dropout) 
                                                                        for i in range(self.layers)])
    def forward(self, x, coordinates=None):
        for layer in self.decoder_layers:
            x = layer(x)
        return x
class MultiHead_Attention_Block3(nn.Module):
    def __init__(self, layers, heads, embeddsize, dropout):
        super(MultiHead_Attention_Block3, self).__init__()
        self.layers = layers
        self.heads = heads
        self.embeddsize = embeddsize
        self.dropout = dropout
        
        self.decoder_layers =  nn.ModuleList([Decoder_Block_2(self.embeddsize, self.heads, self.dropout) 
                                                                        for i in range(self.layers)])
    def forward(self, x, coordinates=None):
        for layer in self.decoder_layers:
            x = layer(x)
        return x
