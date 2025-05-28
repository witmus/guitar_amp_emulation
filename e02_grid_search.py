import torch
import torch.nn as nn
import torchaudio

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from train import train
from models import WindowLSTM
import styles_ranges as sr
from windows import get_sw_paired_dataloader

dry = get_clean_tensor()
crunch = get_crunch_tensor()

train_data_start = sr.SINGLES_RING_OUT_START
train_data_end = sr.SINGLES_RING_OUT_START + 48 * 44100

val_data_start = sr.CHORDS_ARPEGGIO_START
val_data_end = sr.CHORDS_ARPEGGIO_START + 12 * 44100

x_train = dry[train_data_start:train_data_end]
y_train = crunch[train_data_start:train_data_end]

x_val = dry[val_data_start:val_data_end]
y_val = crunch[val_data_start:val_data_end]

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    x_train = x_train.pin_memory()
    y_train = y_train.pin_memory()
    x_val = x_val.pin_memory()
    y_val = y_val.pin_memory()

filter_size = 24
kernel_size = 11
stride = 1

learning_rate = 0.001
epochs = 15
window_size = 2000
batch_size = 600

hidden_nums = [96,128,160,192]
layers_nums = [1,2]

train_dataloader = get_sw_paired_dataloader(x_train,y_train, window_size, batch_size)
val_dataloader = get_sw_paired_dataloader(x_val,y_val, window_size, batch_size)

torch.manual_seed(22150)

for ln,num_layers in enumerate(layers_nums):
    for hn,num_hidden in enumerate(hidden_nums):
        print(f'layers: {num_layers}, hidden: {num_hidden}')
        print()

        scores_path = f'scores/grid_search/gs_lstm_{num_layers}_{num_hidden}.npy'
        model_path = f'models/grid_search/lstm_{num_layers}_{num_hidden}'
        
        model = WindowLSTM(filter_size,kernel_size,stride,num_hidden,num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = train(model,optimizer,train_dataloader,val_dataloader,epochs,device,scores_path,model_path)
        print()
