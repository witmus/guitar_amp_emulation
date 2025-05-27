import torch
import torch.nn as nn
import torchaudio

from audio import get_clean_tensor,get_crunch_tensor
from models import WindowLSTM
import styles_ranges as sr
from train import train
from windows import get_sw_paired_dataloader

dry = get_clean_tensor()
crunch = get_crunch_tensor()

train_data_start = sr.SINGLES_RING_OUT_START
train_data_end = sr.SINGLES_RING_OUT_START + 24 * 44100
# train_data_end = sr.POWER_CHORDS_RING_OUT_END

val_data_start = sr.CHORDS_ARPEGGIO_START
# val_data_end = sr.CHORDS_ARPEGGIO_START + 36 * 44100
val_data_end = sr.CHORDS_ARPEGGIO_START + 6 * 44100

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

n_hidden = 128
n_layers = 1
n_filters = 32
n_strides = 1
kernel_size = 11

learning_rate = 0.001
epochs = 25

batch_sizes = [1000, 1500, 2000]
window_sizes = [200, 400, 600]

torch.manual_seed(22150)

for bs,batch_size in enumerate(batch_sizes):
    for ns,window_size in enumerate(window_sizes): 
        print(f'bs: {batch_size} ns: {window_size}')
        print()
        train_dataloader = get_sw_paired_dataloader(x_train,y_train, window_size, batch_size)
        val_dataloader = get_sw_paired_dataloader(x_val,y_val, window_size, batch_size)

        scores_path = f'scores/batch_sizes/bs_{batch_size}_ns_{window_size}.npy'
        model_path = f'models/batch_sizes/bs_{batch_size}_ns_{window_size}_'

        model = WindowLSTM(n_filters,kernel_size,n_strides,n_hidden,n_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model = train(model,optimizer,train_dataloader,val_dataloader,epochs,device,scores_path,model_path)
        print()
