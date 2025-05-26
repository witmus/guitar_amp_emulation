import torch
import torch.nn as nn

class WindowLSTM(nn.Module):
    def __init__(self, n_conv_outs, s_kernel, n_stride, n_hidden, n_layers):
        super(WindowLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_conv_outs,
            kernel_size=s_kernel,
            stride=n_stride,
            padding='same'
        )

        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(5)
        
        self.conv2 = nn.Conv1d(
            in_channels=n_conv_outs,
            out_channels=n_conv_outs,
            kernel_size=s_kernel,
            stride=n_stride,
            padding='same'
        )

        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(5)
        
        self.lstm = nn.LSTM(
            input_size=n_conv_outs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )

        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x, state=None):
        x = x.permute(0, 2, 1)
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.mp2(self.relu2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, (h_s, c_s) = self.lstm(x, state)
        result = self.out(lstm_out[:,-1,:])
        return result, h_s, c_s
    