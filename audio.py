import torch
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

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(tensor.abs())
    if max_val > 0:
        tensor = tensor / max_val
    return tensor
