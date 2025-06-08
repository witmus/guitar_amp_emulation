import torch
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
import numpy as np

from typing import List

from global_vars import SAMPLE_RATE

def plot_waveform(waveform, sample_rate=SAMPLE_RATE, title='Waveform', channel_titles=['c1', 'c2']):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].set_xlabel('Czas [s]')
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(channel_titles[c])
        else:
            axes[c].set_ylabel('Poziom sygnału')
    figure.suptitle(title)
    plt.show()

def plot_spectrum(waveform, sample_rate=SAMPLE_RATE, title='Spektrum'):
    transform = Spectrogram(800)
    spectrum = transform(waveform)
    fft = torch.fft.fft(spectrum)
    magnitude = torch.abs(fft)
    half_len = len(magnitude) // 2
    magnitude = magnitude[:half_len]
    freqs = torch.linspace(0, sample_rate / 2, half_len)
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs.numpy(), magnitude.numpy())
    plt.title(title)
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Magnitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_stats(losses: List[float]):
    result = np.zeros(shape=(5))
    result[0] = np.mean(losses)
    result[1] = np.std(losses)
    result[2] = np.median(losses)
    result[3] = np.min(losses)
    result[4] = np.max(losses)
    return result
