import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from cv_train_lstm import cv_train_lstm
from train import train
from models import DoubleConvLSTM

learning_rate = 0.001
epochs = 15
num_steps = 100
batch_size = 200

num_layers = 2
num_hidden = 192
filters = 16
strides = 1
kernel_size = 11

folds = 10
repeats = 5
seed = 22150

dry = get_clean_tensor()
dry = normalize_tensor(dry)
crunch = get_crunch_tensor()
crunch = normalize_tensor(crunch)

# data_time_seconds = int(2.5 * 60)
data_time_seconds = 10
data_samples = int(44_100 * data_time_seconds)

x = dry[0][:data_samples].view(folds,-1)
y = crunch[0][:data_samples].view(folds,-1)

device = "cuda" if torch.cuda.is_available() else "cpu"

rkf = RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)

lstm_scores = np.zeros(shape=(folds,epochs,7))
lstm_scores_path = f'scores/wavenet_vs_lstm/lstm.npy'
lstm_model_path = f'models/wavenet_vs_lstm/lstm_'
np.save(lstm_scores_path,lstm_scores)

for i,(train,test) in enumerate(rkf.split(range(folds))):
    print(f'fold: {i}')
    print()

    print('lstm')
    
    lstm_model = DoubleConvLSTM(n_conv_outs=filters,s_kernel=kernel_size,n_stride=strides,n_hidden=num_hidden, n_layers=num_layers)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    lstm_model = cv_train_lstm(lstm_model,lstm_optimizer,x,y,train,test,epochs,batch_size,num_steps,device,lstm_scores_path,lstm_model_path,i)
    print()

    #todo: wavenet loop