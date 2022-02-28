import pytorch_lightning as pl
import torchmetrics as tm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling import MA_Pooling, KNN_Pooling
from embedding_modules import Naive_Embedding,  Neighboorhood_Embedding
from Attention_modules.attention_blocks import Self_Attention_Block, MultiHead_Attention_Block, MultiHead_Attention_Block3

class PCT_Semantic_Segmentation_Network(nn.pl.LightningModule):
    def __init__(self, num_classes, input_features=3, encoder_dimensionality=128, linear_encoder_layer=1024, 
                 segmentation_layer_size = 256, drop_out=0.5, naive_embedding=False, k=32):
        super(PCT_Semantic_Segmentation_Network, self).__init__()
        
        self.input_features = input_features
        self.encoder_dimensionality = encoder_dimensionality
        self.linear_encoder_layer = linear_encoder_layer
        self.segmentation_layer_size = segmentation_layer_size
        self.drop_out = drop_out
        self.num_classes = num_classes
        self.naive_embedding = naive_embedding
        self.k = k
        
     
        #samplingis assigned as false as samplingcannot be used in conjunction with semantic segmentation
        self.neighboorhood_embedding = Neighboorhood_Embedding(self.input_features, self.encoder_dimensionality, self.k, self.naive_embedding)
        self.attention_block = Self_Attention_Block(self.encoder_dimensionality, self.linear_encoder_layer)
            
        self.segmentation_layers = nn.Sequential(
            nn.Conv1d(self.linear_encoder_layer*3, self.segmentation_layer_size*2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.segmentation_layer_size*2),
            nn.ReLU(),
            nn.Dropout(p = self.drop_out),
            nn.Conv1d(self.segmentation_layer_size*2, self.segmentation_layer_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.segmentation_layer_size),
            nn.ReLU(),
            nn.Conv1d(self.segmentation_layer_size, self.num_classes, kernel_size=1, bias=False)
       )

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y.squeeze(1))
        accuracy = tm.functional.accuracy(preds, y)
        self.log("loss", loss, on_epoch=True, prog_bar=True)                         
        self.log("accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y.squeeze(1))
        accuracy = tm.functional.accuracy(preds, y)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
    def forward(self, x):
        x = self.neighboorhood_embedding(x)
        x = self.attention_block(x)
        global_feature = pooling(x)
        x = torch.cat((x, global_feature), dim=1)
        x = segmentation_layers(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
