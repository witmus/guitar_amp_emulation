from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import time

from loss import ESRDCLoss
from utilities import get_stats
from models import WindowLSTM

def train_cv_lstm(
        model: WindowLSTM,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        device: str,
        scores_path: str,
        model_path: str,
        fold: int
    ):
    train_batches = len(train_dataloader)

    loss_fn = ESRDCLoss()
    scores = np.load(scores_path)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        print('epoch: ', epoch)
        losses = []
        epoch_start = time.time()

        epoch_start = time.time()
        for i,(x_t,y_t) in enumerate(train_dataloader):
            y_t = y_t.unsqueeze(1).to(device)
            optimizer.zero_grad()

            pred,_,_ = model(x_t.to(device))
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            print('batch: ', i, '/', train_batches, ' loss: ', loss_value, end='\r')

        epoch_time = time.time() - epoch_start
        epoch_stats = get_stats(losses)

        print('train loss: ', epoch_stats[0])

        torch.cuda.empty_cache()
        model.eval()

        val_losses = []
        
        with torch.no_grad():
            for i,(x_v,y_v) in enumerate(val_dataloader):
                test,_,_ = model(x_v.to(device))
                y_v = y_v.unsqueeze(1).to(device)
                val_losses.append(loss_fn(test,y_v).item())
            
        val_loss = np.mean(val_losses)
        val_std = np.mean(val_losses)
        print('val loss: ', val_loss)
        
        scores[fold,epoch,:5] = epoch_stats
        scores[fold,epoch,5] = val_loss
        scores[fold,epoch,6] = val_std
        scores[fold,epoch,7] = epoch_time
        
        torch.save(model.state_dict(), model_path + f'checkpoint_{fold}_{epoch}.pth')
        np.save(scores_path,scores)

        torch.cuda.empty_cache()

    return model
