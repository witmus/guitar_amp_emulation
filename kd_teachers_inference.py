import numpy as np
import torch
import torchaudio
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor, get_lstm_teacher_tensor, get_wavenet_teacher_tensor
from train import train
from models import WindowLSTM
from wavenet import WaveNet
import styles_ranges as sr
from windows import get_sw_dataloader, get_wavenet_dataloader

lstm_channels = 24
lstm_kernel_size = 11
lstm_stride = 1
lstm_hidden_size = 128
lstm_layers = 1
lstm_window_size = 2000
lstm_batch_size = 600

wn_steps = 4000
wn_batch_size = 5
wn_channels = 32
wn_dilation_depth = 8
wn_repeats = 4
wn_kernel_size = 3
data_start = sr.SINGLES_RING_OUT_START
data_end = sr.SINGLES_RING_OUT_END

data_samples = int(44_100 * (data_end - data_start))
dry = get_clean_tensor()
crunch = get_crunch_tensor()

x = dry[data_start:data_end]
y = crunch[data_start:data_end]

lstm_teacher_path = 'models/knowledge_distillation/teachers/lstm_checkpoint_19.pth'
wavenet_teacher_path = 'models/knowledge_distillation/teachers/wavenet_checkpoint_19.pth'

if torch.cuda.is_available():
    is_cuda = True
    device = "cuda"
else:
    is_cuda = False
    device = "cpu"

lstm_teacher = WindowLSTM(
    n_conv_outs=lstm_channels,
    s_kernel=lstm_kernel_size,
    n_stride=lstm_stride,
    n_hidden=lstm_hidden_size,
    n_layers=lstm_layers
).to(device)

lstm_teacher.load_state_dict(
    torch.load(lstm_teacher_path,weights_only=True)
)

wavenet_teacher = WaveNet(
    num_channels=wn_channels,
    dilation_depth=wn_dilation_depth,
    num_repeat=wn_repeats,
    kernel_size=wn_kernel_size
).to(device)

wavenet_teacher.load_state_dict(
    torch.load(wavenet_teacher_path,weights_only=True)
)

lstm_teacher.eval()
wavenet_teacher.eval()

lstm_dataloader = get_sw_dataloader(x, lstm_window_size, lstm_batch_size, is_cuda)
wavenet_dataloader = get_wavenet_dataloader(x, wn_steps, wn_batch_size, is_cuda)

lstm_result = torch.tensor([])
with torch.no_grad():
    for i,x_t in enumerate(lstm_dataloader):
        print(i,'/',len(lstm_dataloader),end='\r')
        pred,_,_ = lstm_teacher(x_t.to(device))
        lstm_result = torch.concat((lstm_result, pred.detach().flatten()))

print(x.shape)
print(lstm_result.shape)

wavenet_result = torch.tensor([])
with torch.no_grad():
    for i,x_t in enumerate(wavenet_dataloader):
        print(i+1,'/',len(wavenet_dataloader),end='\r')
        pred = wavenet_teacher(x_t.to(device))
        pred = pred[:,:,-wn_steps:]
        wavenet_result = torch.concat((wavenet_result, pred.detach().flatten()))

print(x.shape)
print(wavenet_result.shape)

torchaudio.save('data/samples/processed/wavenet_teacher_result.wav',wavenet_result.unsqueeze(0),44100)
torchaudio.save('data/samples/processed/lstm_teacher_result.wav',lstm_result.unsqueeze(0),44100)