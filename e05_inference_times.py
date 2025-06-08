import numpy as np
import torch
from torch.profiler import profile,record_function,ProfilerActivity, schedule

from audio import get_dry_tensor
from models import WindowLSTM
from wavenet import WaveNet
from windows import get_sw_dataloader, get_wavenet_dataloader

from pprint import pprint

def test_inference(model, dataloader, scores, scores_path, i_m, i_s):
    print('warmup')
    for i,x in enumerate(dataloader):
        print(f'{i+1}/{len(dataloader)}',end='\r')
        _ = model(x)

    repeats = 10
    for r in range(repeats):
        with profile(activities=[ProfilerActivity.CPU],record_shapes=True) as prof:
            with record_function('model_inference'):
                print('repeat: ', r)
                with torch.no_grad():
                    for i,x in enumerate(dataloader):
                        print(f'{i+1}/{len(dataloader)}',end='\r')
                        _ = model(x)
        with open(f'{scores_path}/averages_{i_m}_{i_s}_{r}.txt', mode='w') as f:
            f.write(str(prof.key_averages()))
        scores[i_m,i_s,r] = prof.profiler.self_cpu_time_total
        np.save(scores_path + '/inference_times.npy', scores)


lstm_num_steps = 600
lstm_batch_size = 2000

lstm_small_num_hidden = 64
lstm_big_num_hidden = 128
lstm_biggest_num_hidden = 192

lstm_num_layers = 1
lstm_filters = 24
lstm_strides = 1
lstm_kernel_size = 11

wavenet_steps = 4000
wavenet_batch_size = 5

wavenet_big_channels = 32
wavenet_big_repeats = 4

wavenet_small_channels = 24
wavenet_small_repeats = 2

wavenet_dilation_depth = 8
wavenet_kernel_size = 3

sample_paths = ['frantic', 'sax', 'smoke', 'tornado']

wavenet_small = WaveNet(wavenet_small_channels, wavenet_dilation_depth, wavenet_small_repeats, wavenet_kernel_size, wavenet_steps)
wavenet_small_path = 'models/knowledge_distillation/students/wavenet_from_wavenet_checkpoint_24_14.pth'
wavenet_small.load_state_dict(torch.load(wavenet_small_path, weights_only=True, map_location=torch.device('cpu')))
wavenet_small.eval()

wavenet_big = WaveNet(wavenet_big_channels, wavenet_dilation_depth, wavenet_big_repeats, wavenet_kernel_size, wavenet_steps)
wavenet_big_path = 'models/knowledge_distillation/teachers/wavenet_checkpoint_19.pth'
wavenet_big.load_state_dict(torch.load(wavenet_big_path, weights_only=True, map_location=torch.device('cpu')))
wavenet_big.eval()

lstm_small_path = 'models/knowledge_distillation/students/lstm_from_wavenet_checkpoint_24_14.pth'
lstm_big_path = 'models/grid_search/lstm_1_128checkpoint_14.pth'
lstm_biggest_path = 'models/grid_search/lstm_1_192checkpoint_14.pth'

lstm_small = WindowLSTM(lstm_filters, lstm_kernel_size, lstm_strides, lstm_small_num_hidden, lstm_num_layers)
lstm_small.load_state_dict(torch.load(lstm_small_path, weights_only=True, map_location=torch.device('cpu')))
lstm_small.eval()

lstm_big = WindowLSTM(lstm_filters, lstm_kernel_size, lstm_strides, lstm_big_num_hidden, lstm_num_layers)
lstm_big.load_state_dict(torch.load(lstm_big_path, weights_only=True, map_location=torch.device('cpu')))
lstm_big.eval()

lstm_biggest = WindowLSTM(lstm_filters, lstm_kernel_size, lstm_strides, lstm_biggest_num_hidden, lstm_num_layers)
lstm_biggest.load_state_dict(torch.load(lstm_biggest_path, weights_only=True, map_location=torch.device('cpu')))
lstm_biggest.eval()

models = [wavenet_small, wavenet_big, lstm_small, lstm_big, lstm_biggest]
sample_time_seconds = 8

scores_path = f'scores/inference_times/{sample_time_seconds}s'
scores = np.load(scores_path + '/inference_times.npy')

for m,model in enumerate(models):
    for s,sample_path in enumerate(sample_paths):
        print(m, ' ', sample_path)
        x = get_dry_tensor(sample_path)[:(44100 * sample_time_seconds)]
        dataloader = \
            get_sw_dataloader(x, lstm_num_steps, lstm_batch_size, False) if m > 1 \
            else get_wavenet_dataloader(x, wavenet_steps, wavenet_batch_size, False)
        test_inference(model, dataloader, scores, scores_path, m, s)
