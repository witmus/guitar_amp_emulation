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
        num_layers,
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
    val_padding = samples_per_batch - (X_val.size(0) % samples_per_batch)
    if val_padding < samples_per_batch:
        X_val = torch.concatenate((X_val,torch.zeros(val_padding)))
        Y_val = torch.concatenate((Y_val,torch.zeros(val_padding)))

    # x_val_batched = X_val.view(-1,batch_size,num_steps,1)
    # y_val_batched = Y_val.view(-1,batch_size,num_steps,1)

    train_batches = int(train_samples / samples_per_batch)
    val_batches = int(X_val.size(0) / samples_per_batch)

    loss_fn = ESRDCLoss()

    epochs_losses = []

    scores = np.zeros(shape=(epochs,5))

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        hidden_state = torch.zeros(num_layers,batch_size,hidden_size).to(device)
        cell_state = torch.zeros(num_layers,batch_size,hidden_size).to(device)
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
        del x
        del y
        del hidden_state
        del cell_state
        torch.cuda.empty_cache()

        with torch.no_grad():
            val_losses = []
            val_hidden_state = torch.zeros(num_layers,batch_size,hidden_size).to(device)
            val_cell_state = torch.zeros(num_layers,batch_size,hidden_size).to(device)
            for i,vb in enumerate(range(val_batches)):
                print(i,end='\r')
                val_offset = vb * samples_per_batch
                val_x_batch = X_val[val_offset:val_offset + samples_per_batch].view(batch_size,num_steps,1).to(device)
                val_y_batch = Y_val[val_offset:val_offset + samples_per_batch].view(batch_size,num_steps,1).to(device)
                val_state = (val_hidden_state,val_cell_state)
                test,val_hidden_state,val_cell_state = model(val_x_batch, val_state)
                val_losses.append(loss_fn(test,val_y_batch).item())
            
        val_loss = np.mean(val_losses)
        print('val loss: %f' % val_loss)
        scores[epoch,0] = mean_loss
        scores[epoch,1] = val_loss
        scores[epoch,2] = epoch_time
        # scores[epoch,3] = prof.key_averages().self_cpu_time_total
        # scores[epoch,4] = sum([i.cuda_time for i in prof.key_averages()])
        
        torch.save(model.state_dict(), model_path + f'checkpoint_{epoch}.pth')
        np.save(scores_path,scores)

        torch.cuda.empty_cache()

    return model
