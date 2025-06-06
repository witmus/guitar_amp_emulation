from typing import Sequence
import torch
from torch.utils.data import DataLoader
import torchaudio

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(tensor.abs())
    if max_val > 0:
        tensor = tensor / max_val
    return tensor

def get_dry_tensor(path: str) -> torch.Tensor:
    tensor, _ = torchaudio.load(f'data/samples/clean/{path}_clean.wav')
    return normalize_tensor(tensor[0])

def get_wet_tensor(path: str) -> torch.Tensor:
    tensor, _ = torchaudio.load(f'data/samples/processed/{path}_crunch.wav')
    return normalize_tensor(tensor[0])

def get_clean_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/clean/thesis_data_clean.wav')
    return normalize_tensor(tensor[0])

def get_crunch_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/thesis_data_crunch.wav')
    return normalize_tensor(tensor[0])

def get_distortion_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/thesis_data_dist.wav')
    return normalize_tensor(tensor[0])

def get_lstm_teacher_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/lstm_teacher_result.wav')
    return normalize_tensor(tensor[0])

def get_wavenet_teacher_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/processed/wavenet_teacher_result.wav')
    return normalize_tensor(tensor[0])

def get_frantic_clean_tensor() -> torch.Tensor:
    tensor, _ = torchaudio.load('data/samples/clean/frantic.wav')
    return normalize_tensor(tensor[0])
