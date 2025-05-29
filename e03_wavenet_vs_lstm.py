import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from train_cv_lstm import train_cv_lstm
import styles_ranges as sr
from train_cv_wavenet import train_cv_wavenet
from windows import get_sw_paired_dataloader, get_wavenet_paired_dataloader

from models import WindowLSTM
from wavenet import WaveNet

learning_rate = 0.001
epochs = 10
window_size = 600
batch_size = 2000

num_layers = 1
num_hidden = 64
filters = 24
strides = 1
kernel_size = 11

#100 ms per batch
wn_steps = 4000
wn_batch_size = 5
wn_channels = 32
wn_dilation_depth = 8
wn_repeats = 3
wn_kernel_size = 3

folds = 5
repeats = 5
seed = 22150

dry = get_clean_tensor()
crunch = get_crunch_tensor()

data_start = sr.SINGLES_RING_OUT_START
data_end = sr.SINGLES_RING_OUT_START + 60 * 44100

x = dry[data_start:data_end].view(folds * 2, -1)
y = crunch[data_start:data_end].view(folds * 2, -1)

if torch.cuda.is_available():
    is_cuda = True
    device = "cuda"
else:
    is_cuda = False
    device = "cpu"
    
rkf = RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)

# lstm_scores = np.zeros(shape=(folds*repeats,epochs,8))
lstm_scores_path = f'scores/wavenet_vs_lstm/lstm.npy'
lstm_model_path = f'models/wavenet_vs_lstm/lstm_'
# np.save(lstm_scores_path,lstm_scores)
lstm_scores = np.load(lstm_scores_path)

# wn_scores = np.zeros(shape=(folds*repeats,epochs,8))
wn_scores_path = f'scores/wavenet_vs_lstm/wavenet.npy'
wn_model_path = f'models/wavenet_vs_lstm/wavenet_'
# np.save(wn_scores_path,wn_scores)
wn_scores = np.load(wn_scores_path)

torch.manual_seed(22150)
fold_checkpoint = 3

for i,(train,val) in enumerate(rkf.split(range(folds * 2))):
    print(f'fold: {i}')
    print()
    if i < fold_checkpoint:
        continue
    
    x_train = x[train].flatten()
    y_train = y[train].flatten()

    x_val = x[val].flatten()
    y_val = y[val].flatten()

    print('lstm')
    lstm_train_dataloader = get_sw_paired_dataloader(x_train,y_train, window_size, batch_size, is_cuda)
    lstm_val_dataloader = get_sw_paired_dataloader(x_val,y_val, window_size, batch_size, is_cuda)
    
    lstm_model = WindowLSTM(n_conv_outs=filters,s_kernel=kernel_size,n_stride=strides,n_hidden=num_hidden, n_layers=num_layers)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    lstm_model = train_cv_lstm(lstm_model, lstm_optimizer, lstm_train_dataloader, lstm_val_dataloader, epochs, device, lstm_scores_path, lstm_model_path, i)
    print()
    
    # print("wavenet")
    # wavenet_train_dataloader = get_wavenet_paired_dataloader(x_train,y_train, wn_steps, wn_batch_size, is_cuda)
    # wavenet_val_dataloader = get_wavenet_paired_dataloader(x_val,y_val, wn_steps, wn_batch_size, is_cuda)

    # wavenet_model = WaveNet(wn_channels, wn_dilation_depth, wn_repeats, wn_kernel_size)
    # wavenet_optimizer = torch.optim.Adam(wavenet_model.parameters(), lr=learning_rate)
    # wavenet_model = train_cv_wavenet(wavenet_model, wavenet_optimizer, wavenet_train_dataloader, wavenet_val_dataloader, epochs, device, wn_scores_path, wn_model_path, i)
    
    # print()