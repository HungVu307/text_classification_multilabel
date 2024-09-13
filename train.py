import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from dataset import TextClassificationDataset, LetterBox
from model import mobileone
import os
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_model import TextClassificationLightningModel

torch.set_float32_matmul_precision('medium')

def main():
    with open('config/main.yml', 'r') as file:
        config = yaml.safe_load(file)
    data_dir = config['data']['data_dir']
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
            LetterBox(),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            LetterBox(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {x: TextClassificationDataset(data_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    
    net = mobileone(num_classes=config['model']['num_class'], variant='s1')
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  
        dirpath=config['data']['checkpoint_path'],  
        filename='{epoch}-{val_acc:.4f}',  
        save_top_k=2,  
        mode='max' 
    )
    # Logger tensorboard
    logger = TensorBoardLogger("lightning_logs", name=config['model']['name'])
    
    model = TextClassificationLightningModel(
        model=net, 
        config=config,
        criterion=criterion, 
        learning_rate=config['train']['lr'], 
        num_classes=config['model']['num_class'],
        train_dataset=datasets['train'],
        val_dataset=datasets['val']
    )

    trainer = pl.Trainer(max_epochs=config['train']['num_epoch'], 
                         accelerator='gpu',
                        #  val_check_interval=config['train']['val_check_interval'],
                         logger=logger,
                         log_every_n_steps=1,
                         callbacks = [checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()
