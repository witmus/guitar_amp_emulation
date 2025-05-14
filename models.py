import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )

        self.out = nn.Linear(n_hidden,1)

    def forward(self, x, state=None):
        r_out, (h_s, c_s) = self.lstm(x, state)
        outs = []
        for ts in range(r_out.size(1)):
            outs.append(self.out(r_out[:,ts,:]))

        return torch.stack(outs, dim=1), h_s, c_s

class SingleConvLSTM(nn.Module):
    def __init__(self, n_conv_outs, s_kernel, n_stride, n_hidden, n_layers):
        super(SingleConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_conv_outs,
            kernel_size=s_kernel,
            stride=n_stride,
            padding='same'
        )
        
        self.lstm = nn.LSTM(
            input_size=n_conv_outs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )

        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x, state=None):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_s, c_s) = self.lstm(x, state)
        outs = []
        for ts in range(lstm_out.size(1)):
            outs.append(self.out(lstm_out[:,ts,:]))

        return torch.stack(outs, dim=1), h_s, c_s


class DoubleConvLSTM(nn.Module):
    def __init__(self, n_conv_outs, s_kernel, n_stride, n_hidden, n_layers):
        super(DoubleConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_conv_outs,
            kernel_size=s_kernel,
            stride=n_stride,
            padding='same'
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=n_conv_outs,
            out_channels=n_conv_outs,
            kernel_size=s_kernel,
            stride=n_stride,
            padding='same'
        )
        
        self.lstm = nn.LSTM(
            input_size=n_conv_outs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )

        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x, state=None):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_s, c_s) = self.lstm(x, state)
        outs = []
        for ts in range(lstm_out.size(1)):
            outs.append(self.out(lstm_out[:,ts,:]))

        return torch.stack(outs, dim=1), h_s, c_s
    