from typing import Sequence
import torch
from torch.utils.data import DataLoader
import torchaudio

def get_clean_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/clean/thesis_data_clean.wav')
    return tensor

def get_crunch_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/thesis_data_crunch.wav')
    return tensor

def get_distortion_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/thesis_data_dist.wav')
    return tensor
