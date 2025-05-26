import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from train_cv import train_cv
import styles_ranges as sr
from windows import get_sw_paired_dataloader, get_wavenet_paired_dataloader

from models import WindowLSTM
from wavenet import WaveNet

learning_rate = 0.001
epochs = 15
window_size = tbd
batch_size = tbd

num_layers = tbd
num_hidden = tbd
filters = tbd
strides = 1
kernel_size = 11

#100 ms per batch
wn_steps = 4000
wn_batch_size = 64
wn_channels = tbd
wn_dilation_depth = 8
wn_repeats = 3
wn_kernel_size = 3

folds = 5
repeats = 5
seed = 22150

dry = get_clean_tensor()
crunch = get_crunch_tensor()

data_start = sr.SINGLES_RING_OUT_START
data_end = sr.CHORDS_ARPEGGIO_END

x = dry[data_start:data_end].view(folds,-1)
y = crunch[data_start:data_end].view(folds,-1)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    x = x.pin_memory()
    y = y.pin_memory()
    
rkf = RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)

lstm_scores = np.zeros(shape=(folds*repeats,epochs,7))
lstm_scores_path = f'scores/wavenet_vs_lstm/lstm.npy'
lstm_model_path = f'models/wavenet_vs_lstm/lstm_'
np.save(lstm_scores_path,lstm_scores)

wn_scores = np.zeros(shape=(folds*repeats,epochs,7))
wn_scores_path = f'scores/wavenet_vs_lstm/wavenet.npy'
wn_model_path = f'models/wavenet_vs_lstm/wavenet_'
np.save(wn_scores_path,wn_scores)

for i,(train,val) in enumerate(rkf.split(range(folds))):
    print(f'fold: {i}')
    print()
    x_train = x[train].flatten()
    y_train = y[train].flatten()

    x_val = x[val].flatten()
    y_val = y[val].flatten()

    print('lstm')
    lstm_train_dataloader = get_sw_paired_dataloader(x_train,y_train, window_size, batch_size)
    lstm_val_dataloader = get_sw_paired_dataloader(x_val,y_val, window_size, batch_size)
    
    lstm_model = WindowLSTM(n_conv_outs=filters,s_kernel=kernel_size,n_stride=strides,n_hidden=num_hidden, n_layers=num_layers)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    lstm_model = train_cv(lstm_model, lstm_optimizer, lstm_train_dataloader, lstm_val_dataloader, epochs, device, lstm_scores_path, lstm_model_path, i)
    print()
    
    print("wavenet")
    wavenet_train_dataloader = get_wavenet_paired_dataloader(x_train,y_train, window_size, batch_size)
    wavenet_val_dataloader = get_wavenet_paired_dataloader(x_val,y_val, window_size, batch_size)

    wavenet_model = WaveNet(wn_channels, wn_dilation_depth, wn_repeats, wn_kernel_size)
    wavenet_optimizer = torch.optim.Adam(wavenet_model.parameters(), lr=learning_rate)
    wavenet_model = train_cv(wavenet_model, wavenet_optimizer, wavenet_train_dataloader, wavenet_val_dataloader, epochs, device, wn_scores_path, wn_model_path, i)
    
    print()