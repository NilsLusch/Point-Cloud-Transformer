import torch
import torch.nn as nn
import torch.nn.functional as F

#Scalling with the sqrt of the embedd size otherwise the variance becomes too large
def scaled_self_attention(q, k, v, key_size):
    weight = torch.matmul(q, k)#b,n,n
    weight = F.softmax(weight/math.sqrt(key_size), dim=-1)
    attention = torch.matmul(weight, v) # b, c, n
    return attention
  
  #version from PCT
def pct_scaled_attention(q, k, v):
    weight = torch.matmul(q, k)#b,n,n
    weight = F.softmax(weight, dim=-1)/(1e-9 + weight.sum(dim=1, keepdims=True))
    attention = torch.matmul(weight, v) # b, c, n
    return attention
 
class Offset_Attention_Layer(nn.Module):
    def __init__(self, channels, key_size, value_size):
        super(Offset_Attention_Layer, self).__init__()
        
        self.channels = channels
        self.key_size = key_size
        self.value_size = value_size
        #embeddings for queries/keys/values
        self.q_conv = nn.Conv1d(self.channels, int(self.channels * self.key_size), kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(self.channels, int(self.channels * self.key_size), kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(self.channels, int(self.channels * self.value_size), kernel_size=1, bias=False)
        #output layer equivalent to the feed-forward layer in the Transformer architecture
        self.lbr = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.ReLU()
        )
      
    def forward(self, x, coordinates=None):
        #optional positional embedding to present in the paper but present in the authors github
        if coordinates != None:
            x = x + coordinates
        query = self.q_conv(x)
        key = self.k_conv(x)
        value = self.v_conv(x)
        #Apply the attention algorithm here
        attention = pct_scaled_attention(query.permute(0, 2, 1), key, value.permute(0, 2, 1))
        #Get the offset
        attention = x-attention.permute(0,2,1)
        attention  = self.lbr(attention)
        x = x + attention
        return x
      
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
#Changes made are to account for the format of the data
class Multihead_Attention_Layer(nn.Module):
    def __init__(self, embedding, heads):
        super(Multihead_Attention_Layer, self).__init__()
        self.embedding = embedding
        self.heads = heads
        assert self.embedding%self.heads == 0, "The number of embedding channels must be divisible by the number of heads" 
        self.head_dim =  self.embedding//self.heads

        #Perform the embedding for keys/queries/values within a single layer for efficency
        self.embedding_layer = nn.Conv1d(self.embedding, self.embedding*3, kernel_size=1, bias=False)
        #Layer for the output
        self.out = nn.Conv1d(self.embedding, self.embedding,  kernel_size=1, bias=False)
      
    def forward(self, x):
        #x = [batch/channels/num_points]
        batch_size, channels, num_points = x.size()
        qkv = self.embedding_layer(x)
        #Split the embebedded queries/keys/values into the different heads
        qkv = qkv.permute(0,2,1)
        qkv = qkv.reshape(batch_size, num_points, self.heads, 3*self.head_dim)
        qkv= qkv.permute(0, 2, 1, 3)# [Batch/Head/Numpoints/Embeddingdim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        #Transpose the key to enable matrixmultiplication with the query
        k = k.permute(0,1,3,2)
        #perform attention seperately for each head
        values = scaled_self_attention(q, k, v, self.embedding/self.heads)
        
        #Concatenate the resulting output from the different heads
        values = values.permute(0, 2, 1, 3) # [Batch/Numpoints/Head/Embeddingdim]
        values = values.reshape(batch_size, num_points, channels)
        
        #return it back into a form usable for convolutional layers
        x = values.permute(0,2,1)
        x =  self.out(x)
        return x        
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
      #In large parts based on:
class Decoder_Block(nn.Module):
    def __init__(self, embeddsize, heads, forward_expansion, dropout):
        super(Decoder_Block, self).__init__()
        self.embeddsize = embeddsize
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        
        self.multihead_attention = Multihead_Attention_Layer(self.embeddsize, self.heads)
        
        self.norm1 = nn.BatchNorm1d(self.embeddsize)
        self.norm2 = nn.BatchNorm1d(self.embeddsize)

        nn.LayerNorm
        self.feed_forward = nn.Sequential(
            nn.Conv1d(self.embeddsize, self.embeddsize*self.forward_expansion, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(self.embeddsize*self.forward_expansion, self.embeddsize, kernel_size=1, bias=True),
            nn.Dropout(self.dropout)
        )
       
    def forward(self, x, coordinates=None):
        if coordinates != None:
            x = x + coordinates
        attention = self.multihead_attention(x)
        x = self.norm1(x+attention)
        forward = self.feed_forward(x)
        x = self.norm2(x+forward)
        return x
#Difference between the Encoder blocks is that this encoder block only uses a single linear layer as its feedforward network
class Decoder_Block_2(nn.Module):
    def __init__(self, embeddsize, heads, dropout):
        super(Decoder_Block_2, self).__init__()
        self.embeddsize = embeddsize
        self.heads = heads
        self.dropout = dropout
        
        self.multihead_attention = Multihead_Attention_Layer(self.embeddsize, self.heads)
        
        self.norm1 = nn.BatchNorm1d(self.embeddsize)
        self.norm2 = nn.BatchNorm1d(self.embeddsize)

        nn.LayerNorm
        self.feed_forward = nn.Sequential(
            nn.Conv1d(self.embeddsize, self.embeddsize, kernel_size=1, bias=True),
            nn.ReLU())
       
    def forward(self, x, coordinates=None):
        if coordinates != None:
            x = x + coordinates
        attention = self.multihead_attention(x)
        x = self.norm1(x+attention)
        forward = self.feed_forward(x)
        x = self.norm2(x+forward)
        return x    
      
