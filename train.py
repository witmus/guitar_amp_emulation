import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.profiler import profile,record_function,ProfilerActivity
import numpy as np
import time

from loss import ESRDCLoss
from utilities import get_stats
from models import WindowLSTM

def train(
        model: WindowLSTM,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        device: str,
        scores_path: str,
        model_path: str
    ):

    loss_fn = ESRDCLoss()

    epochs_losses = []

    scores = np.zeros(shape=(epochs,7))

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        
        print('epoch: ', epoch)
        losses = []

        for i,(x_t, y_t) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pred,_,_ = model(x_t.to(device))
            loss = loss_fn(pred, y_t.to(device))
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            print('batch: ',i,' loss: ', loss_value, end='\r')

        epoch_time = time.time() - epoch_start
        epoch_stats = get_stats(losses)
        print('train mean loss: ', epoch_stats[0])
        epochs_losses.append(epoch_stats[0])

        model.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            val_losses = []
            for i,(x_v,y_v) in enumerate(val_dataloader):
                print(i,end='\r')
                test,_,_ = model(x_v.to(device))
                val_losses.append(loss_fn(test,y_v.to(device)).item())
            
        val_loss = np.mean(val_losses)

        print('val loss: ', val_loss)
        scores[epoch,:5] = epoch_stats
        scores[epoch,5] = val_loss
        scores[epoch,6] = epoch_time
        
        torch.save(model.state_dict(), model_path + f'checkpoint_{epoch}.pth')
        np.save(scores_path,scores)

        torch.cuda.empty_cache()

    return model
