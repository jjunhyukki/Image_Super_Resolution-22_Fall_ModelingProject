import torch
import os
import wandb
import pandas as pd
from torch.utils.data import DataLoader

from modules.dataset import CustomDataset, get_transform
from modules.losses import L1Loss
from modules.model import CustomSRModel
from train import train

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(os.getcwd(), 'models', 'data')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train = pd.read_csv(os.path.join(data_dir, 'customTrain.csv'))
    valid = pd.read_csv(os.path.join(data_dir, 'customValid.csv'))

    train_dataset = CustomDataset(df=train,
                                  data_dir=data_dir,
                                  transform=get_transform())
    valid_dataset = CustomDataset(df=valid,
                                  data_dir=data_dir,
                                  transform=get_transform())

    trainloader = DataLoader(train_dataset,
                             batch_size=16,
                             shuffle=True,
                             num_workers=4)
    validloader = DataLoader(valid_dataset,
                             batch_size=16,
                             shuffle=True,
                             num_workers=4)

    model = CustomSRModel(device=device,
                          swinir_pretrain=True,
                          edsr_pretrain=True).to(device)

    criterion = L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5)

    wandb.login()

    train(epochs=30,
          device=device,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          trainloader=trainloader,
          validloader=validloader,
          log_iter=100,
          save_epoch=1,
          log_wandb=True,
          wandb_project="FILL YOUR NAME OF WANDB PROJECT",
          wandb_entity="FILL YOUR NAME OF WANDB ENTITY")
