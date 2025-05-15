from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
import time

from loss import ESRDCLoss
from wavenet import WaveNet

def cv_train_wavenet(
        model: WaveNet,
        optimizer: Optimizer,
        dry: torch.Tensor,
        wet: torch.Tensor,
        train_idxs: List[int],
        validate_idx: int,
        epochs: int,
        batch_size: int,
        num_steps: int,
        device: str,
        scores_path: str,
        model_path: str,
        fold: int
    ):
    
    x_train = dry[train_idxs].flatten()
    y_train = wet[train_idxs].flatten()
    
    x_val = dry[validate_idx].squeeze()
    y_val = wet[validate_idx].squeeze()

    samples_per_batch = batch_size * num_steps
    train_samples = x_train.size(0)
    val_padding = samples_per_batch - (x_val.size(0) % samples_per_batch)
    if val_padding < samples_per_batch:
        x_val = torch.concatenate((x_val,torch.zeros(val_padding)))
        y_val = torch.concatenate((y_val,torch.zeros(val_padding)))
    
    train_batches = int(train_samples / samples_per_batch)
    val_batches = int(x_val.size(0) / samples_per_batch)

    loss_fn = ESRDCLoss()
    scores = np.load(scores_path)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        print('epoch: %d' % epoch)
        losses = []

        for batch in range(train_batches):
            offset = samples_per_batch * batch
            x = x_train[offset:offset + samples_per_batch].view(batch_size,1,num_steps).to(device)
            y = y_train[offset:offset + samples_per_batch].view(batch_size,1,num_steps).to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y[:,:,-pred.size(2) :])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            print('batch: %d' % batch,end='\r')

        epoch_time = time.time() - epoch_start
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        median_loss = np.median(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)

        print('train loss: %f' % mean_loss)

        torch.cuda.empty_cache()
        model.eval()

        val_losses = []
        
        with torch.no_grad():
            for i,vb in enumerate(range(val_batches)):
                print(i,end='\r')
                val_offset = vb * samples_per_batch
                val_x_batch = x_val[val_offset:val_offset + samples_per_batch].view(batch_size,1,num_steps).to(device)
                val_y_batch = y_val[val_offset:val_offset + samples_per_batch].view(batch_size,1,num_steps).to(device)
                test = model(val_x_batch)
                val_losses.append(loss_fn(test,val_y_batch[:,:,-test.size(2) :]).item())
            
        val_loss = np.mean(val_losses)
        print('val loss: %f' % val_loss)
        
        scores[fold,epoch,0] = mean_loss
        scores[fold,epoch,1] = std_loss
        scores[fold,epoch,2] = median_loss
        scores[fold,epoch,3] = min_loss
        scores[fold,epoch,4] = max_loss
        scores[fold,epoch,5] = val_loss
        scores[fold,epoch,6] = epoch_time

        #todo: setup seperate experiment for time inference
        # scores[epoch,7] = prof.key_averages().self_cpu_time_total
        # scores[epoch,8] = sum([i.cuda_time for i in prof.key_averages()])
        
        torch.save(model.state_dict(), model_path + f'checkpoint_{fold}_{epoch}.pth')
        np.save(scores_path,scores)

        torch.cuda.empty_cache()

    return model
