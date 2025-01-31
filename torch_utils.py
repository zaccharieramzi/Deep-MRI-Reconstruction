import torch
from torch import nn

class ConvBlock(torch.nn.Module):
    def __init__(self, n_convs=5, n_filters=16, in_chans=2, out_chans=2):
        super(ConvBlock, self).__init__()
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.in_chans = in_chans
        self.out_chans = out_chans

        first_conv = nn.Sequential(nn.Conv2d(in_chans, n_filters, kernel_size=3, padding=1, bias=False), nn.ReLU())
        simple_convs = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False), nn.ReLU())
            for i in range(n_convs - 2)
        ])
        last_conv = nn.Conv2d(n_filters, out_chans, kernel_size=3, padding=1, bias=False)
        self.overall_convs = nn.Sequential(first_conv, simple_convs, last_conv)

    def forward(self, x):
        y = self.overall_convs(x)
        return y

def replace_values_on_mask_torch(cnn_fft, kspace, mask):
    mask = mask[..., None]
    mask = mask.expand_as(kspace).float()
    anti_mask = 1.0 - mask
    replace_cnn_fft = anti_mask * cnn_fft + kspace
    return replace_cnn_fft
