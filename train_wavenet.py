from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import time

from loss import ESRDCLoss
from utilities import get_stats
from wavenet import WaveNet

def train_wavenet(
        model: WaveNet,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        device: str,
        model_path: str
    ):

    loss_fn = ESRDCLoss()

    model.to(device)
    train_batches = len(train_dataloader)
    for epoch in range(epochs):
        model.train()
        print('epoch: ', epoch)
        losses = []

        epoch_start = time.time()
        for i,(x_t,y_t) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pred = model(x_t.to(device))
            loss = loss_fn(pred, y_t[:,:,-pred.size(2) :].to(device))
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
                print(i,end='\r')
                test = model(x_v.to(device))
                val_losses.append(loss_fn(test,y_v[:,:,-test.size(2) :].to(device)).item())
            
        val_loss = np.mean(val_losses)
        val_std = np.mean(val_losses)
        print('val loss: ', val_loss)

        #todo: setup seperate experiment for time inference
        # scores[epoch,7] = prof.key_averages().self_cpu_time_total
        # scores[epoch,8] = sum([i.cuda_time for i in prof.key_averages()])
        
        torch.save(model.state_dict(), model_path + f'checkpoint_{epoch}.pth')

        torch.cuda.empty_cache()

    return model
