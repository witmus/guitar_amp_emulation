import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.profiler import profile,record_function,ProfilerActivity
import numpy as np
import time

from loss import ESRDCLoss

def train(
        model: nn.Module,
        optimizer: Optimizer,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        num_steps: int,
        hidden_size: int,
        device: str,
        scores_path: str,
        model_path: str
    ):
    if X_train.shape != Y_train.shape:
        raise Exception(f'invalid train tensors: {X_train.shape} {Y_train.shape}')
    if X_val.shape != Y_val.shape:
        raise Exception(f'invalid val tensors: {X_val.shape} {Y_val.shape}')
    
    samples_per_batch = batch_size * num_steps
    train_samples = X_train.size(0)
    val_padding = num_steps - (X_val.size(0) % num_steps)
    if val_padding < num_steps:
        X_val = torch.concatenate((X_val,torch.zeros(val_padding)))
        Y_val = torch.concatenate((Y_val,torch.zeros(val_padding)))

    x_val_batched = X_val.view(-1,num_steps,1)
    y_val_batched = Y_val.view(-1,num_steps,1)
    val_batches = x_val_batched.size(0)

    train_batches = int(train_samples / samples_per_batch)

    loss_fn = ESRDCLoss()

    epochs_losses = []

    model.to(device=device)

    scores = np.zeros(shape=(epochs,5))

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        hidden_state = torch.zeros(1,batch_size,hidden_size)
        cell_state = torch.zeros(1,batch_size,hidden_size)
        print('epoch: %d' % epoch)
        losses = []

        for batch in range(train_batches):
            offset = samples_per_batch * batch
            x = X_train[offset:offset + samples_per_batch].view(batch_size,num_steps,1).to(device)
            y = Y_train[offset:offset + samples_per_batch].view(batch_size,num_steps,1).to(device)
            
            optimizer.zero_grad()
            state = (hidden_state,cell_state)
            pred,h_s,c_s = model(x,state)
            hidden_state = h_s.data
            cell_state = c_s.data
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            print('batch: %d' % batch,end='\r')

        epoch_time = time.time() - epoch_start
        mean_loss = np.mean(losses)
        print('train loss: %f' % mean_loss)
        epochs_losses.append(mean_loss)

        model.eval()
        val_hidden_state = torch.zeros(1,val_batches,hidden_size)
        val_cell_state = torch.zeros(1,val_batches,hidden_size)
        val_state = (val_hidden_state,val_cell_state)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                test,_,_ = model(x_val_batched,val_state)

        val_loss = loss_fn(test,y_val_batched)

        torch.save(model.state_dict(), model_path + f'checkpoint_{epoch}.pth')

        print('train loss: %f' % mean_loss)
        scores[epoch,0] = mean_loss
        scores[epoch,1] = val_loss.item()
        scores[epoch,2] = epoch_time
        scores[epoch,3] = prof.key_averages().self_cpu_time_total
        if device == 'cuda':
            scores[epoch,4] = sum([item.cuda_time for item in prof.key_averages()])
        
        np.save(scores_path,scores)

    return model
