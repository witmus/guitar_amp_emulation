import torch
import torchaudio.functional as taf
from global_vars import PRE_EMPHASIS_COEFF

class ESRLoss(torch.nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()

    def forward(self, pred, target):
        pred = taf.preemphasis(pred, PRE_EMPHASIS_COEFF)
        target = taf.preemphasis(target, PRE_EMPHASIS_COEFF)
        loss = torch.add(target, -pred)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        loss = torch.div(loss, torch.mean(torch.pow(target, 2)))
        return loss

class DCLoss(torch.nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
    
    def forward(self, pred, target):
        pred = taf.preemphasis(pred, PRE_EMPHASIS_COEFF)
        target = taf.preemphasis(target, PRE_EMPHASIS_COEFF)
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(pred, 0)), 2)
        loss = torch.mean(loss)
        loss = torch.div(loss, torch.mean(torch.pow(target, 2)))
        return loss
    
class ESRDCLoss(torch.nn.Module):
    def __init__(self, esr_weight=0.8):
        super(ESRDCLoss, self).__init__()
        self.esr = ESRLoss()
        self.dc = DCLoss()
        self.esr_weight = esr_weight

    def forward(self, pred, target):
        loss = torch.mul(self.esr(pred, target), self.esr_weight)
        loss += torch.mul(self.dc(pred, target), 1 - self.esr_weight)
        return loss
    
class ESRDistillationLoss(torch.nn.Module):
    def __init__(self, esr_weight=0.8, alpha=0.8):
        super(ESRDistillationLoss, self).__init__()
        self.esr = ESRLoss()
        self.dc = DCLoss()
        self.esr_weight = esr_weight
        self.alpha = alpha

    def forward(self, pred, target, teacher):
        loss = torch.mul(self.esr(pred, target), self.esr_weight)
        loss += torch.mul(self.dc(pred, target), 1 - self.esr_weight)
        loss *= (1 - self.alpha)
        loss += (torch.mul(self.esr(pred, target), self.esr_weight) + torch.mul(self.dc(pred, target), 1 - self.esr_weight)) * self.alpha
        return loss
    