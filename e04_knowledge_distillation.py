import numpy as np
import torch
from sklearn.model_selection import RepeatedKFold

from audio import get_clean_tensor,get_crunch_tensor, get_lstm_teacher_tensor, get_wavenet_teacher_tensor
from train import train
from models import WindowLSTM
from wavenet import WaveNet
import styles_ranges as sr
from windows import get_sw_teacher_dataloader, get_wavenet_teacher_dataloader, get_sw_paired_dataloader, get_wavenet_paired_dataloader

from distill import distill 

learning_rate = 0.001
epochs = 15

lstm_num_steps = tbd
lstm_batch_size = tbd

lstm_student_num_layers = tbd
lstm_student_num_hidden = tbd
lstm_student_filters = tbd
lstm_student_strides = 1
lstm_student_kernel_size = tbd

wavenet_steps = 4000
wavenet_batch_size = 64

wavenet_student_channels = tbd
wavenet_student_dilation_depth = 8
wavenet_student_repeats = 3
wavenet_student_kernel_size = 3

folds = 5
repeats = 5
seed = 22150

dry = get_clean_tensor()
crunch = get_crunch_tensor()

data_start = sr.POWER_CHORDS_RING_OUT_START
data_end = sr.PALM_MUTING_END

device = "cuda" if torch.cuda.is_available() else "cpu"

x = dry[data_start:data_end]
y = crunch[data_start:data_end]

t_lstm = get_lstm_teacher_tensor()
t_wn = get_wavenet_teacher_tensor()

t_lstm = torch.concat((torch.zeros(y.size(0) - t_lstm.size(0)), t_lstm)).view(folds,-1)
t_wn = torch.concat((torch.zeros(y.size(0) - t_wn.size(0)), t_wn)).view(folds,-1)

x = x.view(folds,-1)
y = y.view(folds,-1)

if device == "cuda":
    x = x.pin_memory()
    y = y.pin_memory()
    t_lstm = t_lstm.pin_memory()
    t_wn = t_wn.pin_memory()

rkf = RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)

lstm_from_lstm_scores = np.zeros(shape=(folds*repeats,epochs,16))
lstm_from_lstm_scores_path = f'scores/knowledge_distillation/lstm_from_lstm.npy'
lstm_from_lstm_model_path = f'models/knowledge_distillation/students/lstm_from_lstm'
np.save(lstm_from_lstm_scores_path, lstm_from_lstm_scores)

wavenet_from_wavenet_scores = np.zeros(shape=(folds*repeats,epochs,16))
wavenet_from_wavenet_scores_path = f'scores/knowledge_distillation/wavenet_from_wavenet.npy'
wavenet_from_wavenet_model_path = f'models/knowledge_distillation/students/wavenet_from_wavenet'
np.save(wavenet_from_wavenet_scores_path, wavenet_from_wavenet_scores)

lstm_from_wavenet_scores = np.zeros(shape=(folds*repeats,epochs,16))
lstm_from_wavenet_scores_path = f'scores/knowledge_distillation/lstm_from_wavenet.npy'
lstm_from_wavenet_model_path = f'models/knowledge_distillation/students/lstm_from_wavenet'
np.save(lstm_from_wavenet_scores_path, lstm_from_wavenet_scores)

wavenet_from_lstm_scores = np.zeros(shape=(folds*repeats,epochs,16))
wavenet_from_lstm_scores_path = f'scores/knowledge_distillation/wavenet_from_lstm.npy'
wavenet_from_lstm_model_path = f'models/knowledge_distillation/students/wavenet_from_lstm'
np.save(wavenet_from_lstm_scores_path, wavenet_from_lstm_scores)

torch.manual_seed(22150)

for i,(train,val) in enumerate(rkf.split(range(folds))):
    print(f'fold: {i}')
    print()

    x_train = x[train].flatten()
    y_train = y[train].flatten()

    x_val = x[val].flatten()
    y_val = y[val].flatten()

    tl = t_lstm[train].flatten()
    tw = t_wn[train].flatten()
    
    print('lstm from lstm')

    lstm_train_dataloader = get_sw_teacher_dataloader(x_train, y_train, tl, lstm_num_steps, lstm_batch_size)
    lstm_val_dataloader = get_sw_paired_dataloader(x_val, y_val, lstm_num_steps, lstm_batch_size)
    
    lstm_student_model = WindowLSTM(
        n_conv_outs=lstm_student_filters,
        s_kernel=lstm_student_kernel_size,
        n_stride=lstm_student_strides,
        n_hidden=lstm_student_num_hidden,
        n_layers=lstm_student_num_layers
    )

    lstm_student_optimizer = torch.optim.Adam(lstm_student_model.parameters(), lr=learning_rate)
    lstm_student_model = distill(
        student=lstm_student_model,
        optimizer=lstm_student_optimizer,
        train_loader=lstm_train_dataloader,
        val_loader=lstm_val_dataloader,
        epochs=epochs,
        device=device,
        scores_path=lstm_from_lstm_scores_path,
        model_path=lstm_from_lstm_model_path,
        fold=i
    )
    
    print()
    print('wavenet from wavenet')

    wavenet_train_dataloader = get_wavenet_teacher_dataloader(x_train, y_train, tw, wavenet_steps, wavenet_batch_size)
    wavenet_val_dataloader = get_wavenet_paired_dataloader(x_val, y_val, wavenet_steps, wavenet_batch_size)
    
    wavenet_student_model = WaveNet(
        num_channels=wavenet_student_channels,
        dilation_depth=wavenet_student_dilation_depth,
        num_repeat=wavenet_student_repeats,
        kernel_size=wavenet_student_kernel_size
    )

    wavenet_student_optimizer = torch.optim.Adam(wavenet_student_model.parameters(), lr=learning_rate)
    wavenet_student_model = distill(
        student=wavenet_student_model,
        optimizer=wavenet_student_optimizer,
        train_loader=wavenet_train_dataloader,
        val_loader=wavenet_val_dataloader,
        epochs=epochs,
        device=device,
        scores_path=wavenet_from_wavenet_scores_path,
        model_path=wavenet_from_wavenet_model_path,
        fold=i
    )

    print()
    print('lstm from wavenet')

    lstm_from_wavenet_train_dataloader = get_sw_teacher_dataloader(x_train, y_train, tw, lstm_num_steps, lstm_batch_size)
    
    lstm_student_model = WindowLSTM(
        n_conv_outs=lstm_student_filters,
        s_kernel=lstm_student_kernel_size,
        n_stride=lstm_student_strides,
        n_hidden=lstm_student_num_hidden,
        n_layers=lstm_student_num_layers
    )

    lstm_student_optimizer = torch.optim.Adam(lstm_student_model.parameters(), lr=learning_rate)
    lstm_student_model = distill(
        student=lstm_student_model,
        optimizer=lstm_student_optimizer,
        train_loader=lstm_from_wavenet_train_dataloader,
        val_loader=lstm_val_dataloader,
        epochs=epochs,
        device=device,
        scores_path=lstm_from_wavenet_scores_path,
        model_path=lstm_from_wavenet_model_path,
        fold=i
    )
    
    print()
    print('wavenet from lstm')

    wavenet_from_lstm_train_dataloader = get_wavenet_teacher_dataloader(x_train, y_train, tl, wavenet_steps, wavenet_batch_size)
    
    wavenet_student_model = WaveNet(
        num_channels=wavenet_student_channels,
        dilation_depth=wavenet_student_dilation_depth,
        num_repeat=wavenet_student_repeats,
        kernel_size=wavenet_student_kernel_size
    )

    wavenet_student_optimizer = torch.optim.Adam(wavenet_student_model.parameters(), lr=learning_rate)
    wavenet_student_model = distill(
        student=wavenet_student_model,
        optimizer=wavenet_student_optimizer,
        train_loader=wavenet_from_lstm_train_dataloader,
        val_loader=wavenet_val_dataloader,
        epochs=epochs,
        device=device,
        scores_path=wavenet_from_lstm_scores_path,
        model_path=wavenet_from_lstm_model_path,
        fold=i
    )

    print()
