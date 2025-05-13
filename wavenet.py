# https://github.com/teddykoker/pedalnet

import torch
import torch.nn as nn

def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            nn.Conv1d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        self.convs_sigm = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.convs_tanh = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        out = x
        skips = []

        for conv_sigm, conv_tanh, residual in zip(
            self.convs_sigm, self.convs_tanh, self.residuals
        ):
            x = out
            out_sigm, out_tanh = conv_sigm(x), conv_tanh(x)
            # gated activation
            out = torch.tanh(out_tanh) * torch.sigmoid(out_sigm)
            skips.append(out)
            out = residual(out)
            out = out + x[:, :, -out.size(2) :]  # fit input with layer output

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out
    