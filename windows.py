from typing import Sequence
import torch
from torch.utils.data import DataLoader

class WindowArray(Sequence):
    def __init__(self, x, window_len):
        self.x = x
        self.window_len = window_len
        
    def __len__(self):
        return len(self.x) - self.window_len + 1
    
    def __getitem__(self, index):
        return self.x[index : index + self.window_len].view(self.window_len,-1)

class WindowPairedArray(Sequence):
    def __init__(self, x, y, window_len):
        self.x = x
        self.y = y[window_len-1:] 
        self.window_len = window_len
        
    def __len__(self):
        return len(self.x) - self.window_len + 1
    
    def __getitem__(self, index):
        x_out = self.x[index : index + self.window_len].view(self.window_len,-1)
        y_out = self.y[index]
        return x_out, y_out
    
class WavenetWindowArray(Sequence):
    def __init__(self, x, window_len):
        self.window_len = window_len
        if len(x) % window_len != 0:
            padding = torch.zeros(window_len - (len(x) % window_len))
            x = torch.concat((padding, x))

        self.x = x
    
    def __len__(self):
        return int(len(self.x) / self.window_len) - 1
    
    def __getitem__(self, index):
        x_out = self.x[index * self.window_len : index * self.window_len + 2 * self.window_len].view(-1,2 * self.window_len)
        return x_out
    
class WavenetPairedWindowArray(Sequence):
    def __init__(self, x, y, window_len):
        assert x.shape == y.shape
        self.window_len = window_len
        if len(x) % window_len != 0:
            padding = torch.zeros(window_len - (len(x) % window_len))
            x = torch.concat((padding, x))
            y = torch.concat((padding, y))

        self.x = x
        self.y = y[window_len:]
    
    def __len__(self):
        return int(len(self.x) / self.window_len) - 1
    
    def __getitem__(self, index):
        idx = index * self.window_len
        x_out = self.x[idx : idx + 2 * self.window_len].view(-1,2 * self.window_len)
        y_out = self.y[idx : idx + self.window_len]
        return x_out, y_out
    

def get_sw_dataloader(x: torch.Tensor, window_length: int, batch_size: int) -> DataLoader:
    return DataLoader(WindowArray(x, window_length), batch_size)

def get_sw_paired_dataloader(x: torch.Tensor, y: torch.Tensor, window_length: int, batch_size: int) -> DataLoader:
    return DataLoader(WindowPairedArray(x, y, window_length), batch_size)

def get_wavenet_dataloader(x: torch.Tensor, window_length: int, batch_size: int) -> DataLoader:
    return DataLoader(WavenetWindowArray(x, window_length), batch_size)

def get_wavenet_paired_dataloader(x: torch.Tensor, y: torch.Tensor, window_length: int, batch_size: int) -> DataLoader:
    return DataLoader(WavenetPairedWindowArray(x, y, window_length), batch_size)