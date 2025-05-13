import torch
import torch.nn as nn
import torchaudio

from audio import get_clean_tensor,get_crunch_tensor,get_distortion_tensor,normalize_tensor
from train import train
from models import LSTM

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

n_hidden = 128
n_layers = 1
learning_rate = 0.001
epochs = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# batch_sizes = [100,200,300,400,500]
# steps_nums = [300]

batch_sizes = [500]
steps_nums = [100,200,300]

for bs,batch_size in enumerate(batch_sizes):
    for ns,num_steps in enumerate(steps_nums): 
        print(f'bs: {batch_size} ns: {num_steps}')
        print()
        scores_path = f'scores/batch_sizes/bs_{batch_size}_ns_{num_steps}.npy'
        model_path = f'models/batch_sizes/bs_{batch_size}_ns_{num_steps}_'
        model = LSTM(n_hidden=n_hidden, n_layers=n_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = train(model,optimizer,x_train,y_train,x_val,y_val,epochs,batch_size,num_steps,n_hidden,device,scores_path,model_path)
        print()
