import torch
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


class TextClassificationLightningModel(pl.LightningModule):
    def __init__(self, 
                 model, 
                 config=None, 
                 criterion=None, 
                 learning_rate=None, 
                 num_classes=None, 
                 train_dataset=None, 
                 val_dataset=None):
        super(TextClassificationLightningModel, self).__init__()
        self.model = model
        self.config = config
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = (torch.sigmoid(outputs) >= self.config['train']['threshold']).float()
        acc = (labels == preds).all(dim=1).float().mean()
        self.train_loss.append(loss.item())
        self.train_acc.append(acc.item())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_loss', loss)
        # self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = (torch.sigmoid(outputs) >= self.config['train']['threshold']).float()
        acc = (labels == preds).all(dim=1).float().mean()

        self.val_loss.append(loss.item())
        self.val_acc.append(acc.item())
        # self.log('val_loss', loss)
        # self.log('val_acc', acc)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, epochs=self.trainer.max_epochs, steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=4)
    
    "evaluation phase"
    def predict(self, inputs):
        self.eval()  
        with torch.no_grad(): 
            outputs = self(inputs)
            preds = torch.sigmoid(outputs) >= 0.5
        return preds.float()