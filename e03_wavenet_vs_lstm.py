import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from cv_train_lstm import cv_train_lstm
from cv_train_wavenet import cv_train_wavenet
from train import train
from models import DoubleConvLSTM
from wavenet import WaveNet

learning_rate = 0.001
epochs = 15
num_steps = 100
batch_size = 200

num_layers = 2
num_hidden = 192
filters = 16
strides = 1
kernel_size = 11

#100 ms per batch
wn_steps = 4410
wn_batch_size = 64
wn_channels = 16
wn_dilation_depth = 8
wn_repeats = 3
wn_kernel_size = 3

folds = 10
repeats = 5
seed = 22150

dry = get_clean_tensor()
dry = normalize_tensor(dry)
crunch = get_crunch_tensor()
crunch = normalize_tensor(crunch)

data_time_seconds = int(2.5 * 60)
data_samples = int(44_100 * data_time_seconds)

x = dry[0][:data_samples].view(folds,-1)
y = crunch[0][:data_samples].view(folds,-1)

device = "cuda" if torch.cuda.is_available() else "cpu"

rkf = RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)

# lstm_scores = np.zeros(shape=(folds*repeats,epochs,7))
# lstm_scores_path = f'scores/wavenet_vs_lstm/lstm.npy'
# lstm_model_path = f'models/wavenet_vs_lstm/lstm_'
# np.save(lstm_scores_path,lstm_scores)

wn_scores = np.zeros(shape=(folds*repeats,epochs,7))
wn_scores_path = f'scores/wavenet_vs_lstm/wavenet.npy'
wn_model_path = f'models/wavenet_vs_lstm/wavenet_'
np.save(wn_scores_path,wn_scores)

for i,(train,test) in enumerate(rkf.split(range(folds))):
    print(f'fold: {i}')
    print()

    # print('lstm')
    
    # lstm_model = DoubleConvLSTM(n_conv_outs=filters,s_kernel=kernel_size,n_stride=strides,n_hidden=num_hidden, n_layers=num_layers)
    # lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    # lstm_model = cv_train_lstm(lstm_model,lstm_optimizer,x,y,train,test,epochs,batch_size,num_steps,device,lstm_scores_path,lstm_model_path,i)
    # print()
    
    print("wavenet")

    wavenet_model = WaveNet(wn_channels, wn_dilation_depth, wn_repeats, wn_kernel_size)
    wavenet_optimizer = torch.optim.Adam(wavenet_model.parameters(), lr=learning_rate)
    wavenet_model = cv_train_wavenet(wavenet_model,wavenet_optimizer,x,y,train,test,epochs,wn_batch_size,wn_steps,device,wn_scores_path,wn_model_path,i)
    
    print()