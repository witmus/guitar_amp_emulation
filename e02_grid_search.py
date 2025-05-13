import torch
import torch.nn as nn
import torchaudio

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from train import train
from models import LSTM, SingleConvLSTM, DoubleConvLSTM

dry = get_clean_tensor()
dry = normalize_tensor(dry)
crunch = get_crunch_tensor()
crunch = normalize_tensor(crunch)

train_time_seconds = 2 * 60
val_time_seconds = 0.5 * 60
train_samples = int(44_100 * train_time_seconds)
val_samples = int(44_100 * val_time_seconds)

x = dry[0]
y = crunch[0]

x_train = x[:train_samples]
y_train = y[:train_samples]

x_val = x[train_samples:train_samples+val_samples]
y_val = y[train_samples:train_samples+val_samples]

learning_rate = 0.001
epochs = 25
num_steps = 100
batch_size = 200
filters = 16
strides = 1
kernel_size = 11
device = "cuda" if torch.cuda.is_available() else "cpu"

hidden_nums = [64,128,256]
layers_nums = [1,2]

for ln,num_layers in enumerate(layers_nums):
    for hn,num_hidden in enumerate(hidden_nums):
        print(f'layers: {num_layers}, hidden: {num_hidden}')
        print()

        print('lstm')
        scores_path = f'scores/grid_search/gs_lstm_{num_layers}_{num_hidden}.npy'
        model_path = f'models/grid_search/lstm_{num_layers}_{num_hidden}'
        
        lstm_model = LSTM(n_hidden=num_hidden, n_layers=num_layers)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
        lstm_model = train(lstm_model,lstm_optimizer,x_train,y_train,x_val,y_val,epochs,batch_size,num_steps,num_hidden,num_layers,device,scores_path,model_path)
        print()

        print('single conv')
        scores_path = f'scores/grid_search/gs_sc_{num_layers}_{num_hidden}.npy'
        model_path = f'models/grid_search/sc_{num_layers}_{num_hidden}_'
        
        sc_model = SingleConvLSTM(n_conv_outs=filters, s_kernel=kernel_size, n_stride=strides, n_hidden=num_hidden, n_layers=num_layers)
        sc_optimizer = torch.optim.Adam(sc_model.parameters(), lr=learning_rate)
        sc_model = train(sc_model, sc_optimizer, x_train, y_train, x_val, y_val, epochs, batch_size, num_steps, num_hidden,num_layers, device, scores_path, model_path)
        print()

        print('double conv')
        scores_path = f'scores/grid_search/gs_dc_{num_layers}_{num_hidden}.npy'
        model_path = f'models/grid_search/dc_{num_layers}_{num_hidden}_'
        
        dc_model = DoubleConvLSTM(n_conv_outs=filters, s_kernel=kernel_size, n_stride=strides, n_hidden=num_hidden, n_layers=num_layers)
        dc_optimizer = torch.optim.Adam(dc_model.parameters(), lr=learning_rate)
        dc_model = train(dc_model, dc_optimizer, x_train, y_train, x_val, y_val, epochs, batch_size, num_steps, num_hidden,num_layers, device, scores_path, model_path)
        print()
