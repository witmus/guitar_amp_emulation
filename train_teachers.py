import torch
import torchaudio

from audio import get_clean_tensor, get_crunch_tensor
from windows import get_sw_paired_dataloader, get_wavenet_paired_dataloader
from train import train
from train_wavenet import train_wavenet
import styles_ranges as sr

from models import WindowLSTM
from wavenet import WaveNet

dry = get_clean_tensor()
wet = get_crunch_tensor()

train_start = sr.SINGLES_RING_OUT_START
train_end = sr.SINGLES_RING_OUT_END

val_start = sr.PENTATONIC_SLOW_START
val_end = sr.PENTATONIC_SLOW_START + 20 * 44100

x_train = dry[train_start:train_end]
y_train = wet[train_start:train_end]

x_val = dry[val_start:val_end]
y_val = wet[val_start:val_end]

if torch.cuda.is_available():
    is_cuda = True
    device = "cuda"
else:
    is_cuda = False
    device = "cpu"

learning_rate = 0.001
epochs = 15

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

lstm_train_dataloader = get_sw_paired_dataloader(x_train,y_train, lstm_window_size, lstm_batch_size, is_cuda)
lstm_val_dataloader = get_sw_paired_dataloader(x_val,y_val, lstm_window_size, lstm_batch_size, is_cuda)

wn_train_dataloader = get_wavenet_paired_dataloader(x_train,y_train, wn_steps, wn_batch_size, is_cuda)
wn_val_dataloader = get_wavenet_paired_dataloader(x_val, y_val, wn_steps, wn_batch_size, is_cuda)

torch.manual_seed(22150)

print('wavenet')
wavenet = WaveNet(wn_channels, wn_dilation_depth, wn_repeats, wn_kernel_size, wn_steps)
wavenet_optimizer = torch.optim.Adam(wavenet.parameters(), learning_rate)
train_wavenet(wavenet, wavenet_optimizer, wn_train_dataloader, wn_val_dataloader, epochs, device, 'models/knowledge_distillation/teachers/wavenet_')

print('lstm')
lstm = WindowLSTM(lstm_channels, lstm_kernel_size, lstm_stride, lstm_hidden_size, lstm_layers)
lstm_optimizer = torch.optim.Adam(lstm.parameters(), learning_rate)
train(lstm, lstm_optimizer, lstm_train_dataloader, lstm_val_dataloader, epochs, device, 'temporary/lstm_teacher.npy', 'models/knowledge_distillation/teachers/lstm_')
print()
