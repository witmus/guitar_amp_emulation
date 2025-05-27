import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from loss import ESRDCLoss
from utilities import get_stats

def distill(
        student: torch.nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: str,
        scores_path: str,
        model_path: str,
        fold: int
):
    student.to(device)
    
    loss_fn = ESRDCLoss()
    scores = np.load(scores_path)
    alpha = 0.8
    beta = 0.2

    for epoch in range(epochs):
        ground_truth_losses = []
        distillation_losses = []
        total_losses = []
        student.train()
        print('epoch: ', epoch)
        for i,(x_t,y_t,t_t) in enumerate(train_loader):
            print('batch: ', i, end='\r')
            optimizer.zero_grad()
            pred,_,_ = student(x_t.to(device))

            distillation_loss = loss_fn(pred,t_t.to(device)) * alpha
            ground_truth_loss = loss_fn(pred,y_t.to(device)) * beta
            loss = distillation_loss + ground_truth_loss
            loss.backward()
            optimizer.step()

            ground_truth_losses.append(ground_truth_loss.item())
            distillation_losses.append(distillation_loss.item())
            total_losses.append(loss.item())

        d_loss_stats = get_stats(distillation_losses)
        gt_loss_stats = get_stats(ground_truth_losses)
        total_loss_stats = get_stats(total_losses)

        print('train total loss: ', total_loss_stats[0])

        torch.cuda.empty_cache()
        student.eval()

        val_losses = []
        
        with torch.no_grad():
            for i,(x_v,y_v) in enumerate(val_loader):
                val_pred,_,_ = student(x_v.to(device))
                val_loss = loss_fn(val_pred, y_v.to(device))
                val_losses.append(val_loss.item())
            
        val_mean_loss = np.mean(val_losses)
        print('val loss: ', val_mean_loss)
        
        scores[fold,epoch,0:5] = d_loss_stats
        scores[fold,epoch,5:10] = gt_loss_stats
        scores[fold,epoch,10:15] = total_loss_stats

        scores[fold,epoch,15] = val_mean_loss
        
        torch.save(student.state_dict(), model_path + f'_checkpoint_{fold}_{epoch}.pth')
        np.save(scores_path,scores)

        torch.cuda.empty_cache()

    return student
