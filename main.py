import torch
import torch.nn as nn
import torchaudio

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from utilities import plot_waveform

dry = get_clean_tensor()
dry = normalize_tensor(dry)
# plot_waveform(dry)
crunch = get_crunch_tensor()
# plot_waveform(crunch)
distortion = get_distortion_tensor()
# plot_waveform(distortion)

train_time_seconds = 0.1 * 60
val_time_seconds = 0.01 * 60
train_samples = int(44_100 * train_time_seconds)
val_samples = int(44_100 * val_time_seconds)

x = dry[0]
y = crunch[0]

x_train = x[:train_samples]
y_train = y[:train_samples]

x_val = x[train_samples:train_samples+val_samples]
y_val = y[train_samples:train_samples+val_samples]
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
from train import train
from models import LSTM
model = LSTM(n_hidden=64,n_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
try:
    model = train(model,optimizer,x_train,y_train,x_val,y_val,10,100,200,64,'cuda','temporary/temp','temporary/')
except Exception as e:
    print(e)