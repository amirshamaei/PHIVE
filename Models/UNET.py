from torch import nn
import torch

class BasicBlock_DOWN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size= 5,
        padding = 2,
        stride=1,
        dropout=0,
        **kwargs,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(4,out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(4,out_channels)
        self.act = nn.LeakyReLU()

        self.shortcut = nn.Identity()
        self.downsample = nn.Conv1d(in_channels,out_channels,kernel_size=2, stride = 2, bias=False)
    def forward(self, x):
        out = self.downsample(x)
        shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out

class ConvNet_ENC(nn.Module):
    @staticmethod
    def calc_activation_shape(
            dim, ksize, dilation=1, stride=1, padding=0
    ):
        def shape_each_dim():
            odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
            return (odim_i / stride) + 1

        return shape_each_dim()
    def __init__(self, in_chanel, latent_Size,dropout=0,freeze_enc=False,depth=5):
        super().__init__()
        layers = []
        init = nn.Sequential(
            nn.Conv1d(in_chanel, 16, 5, stride=2, padding=2),
            nn.Conv1d(16, 16, 5, stride=2, padding=2))
        layers.append(init)
        out_channels = [16, 16, 32, 64,128,256]

        for i in range(depth-1):
            layers.append(BasicBlock_DOWN(out_channels[i],out_channels[i+1],dropout=dropout))
        self.encoder = nn.Sequential(*layers)

        # self.attention = nn.MultiheadAttention(out_channels[i+1], 2,batch_first=True,bias=False)

        self.reg = nn.Sequential(nn.Flatten(),
                                       nn.LazyLinear(latent_Size))
        if freeze_enc==True:
            self.encoder.requires_grad_(False)

    def forward(self, x):
        x_ = self.encoder(x)
        # x_ = self.attention(x_.transpose(2,1),x_.transpose(2,1),x_.transpose(2,1))[0].transpose(2,1)
        return x_

    def regres(self,x_):
        return self.reg(x_)


class BasicBlock_UP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            stride=1,
            dropout = 0,
            **kwargs,
    ):
        super().__init__()

        self.conv1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(4,out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(4,out_channels)
        self.act = nn.LeakyReLU()

        self.shortcut = nn.Identity()
        self.upsample = nn.ConvTranspose1d(in_channels,out_channels,kernel_size=2, stride = 2, bias=False)

    def forward(self, x):
        out = self.upsample(x)
        shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out


class ConvNet_DEC(nn.Module):
    @staticmethod
    def calc_activation_shape(
            dim, ksize, dilation=1, stride=1, padding=0
    ):
        def shape_each_dim():
            odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
            return (odim_i / stride) + 1

        return shape_each_dim()

    def __init__(self,out_chanel,dropout=0,freeze_dec=False):
        super().__init__()
        layers = []
        out_channels = [128, 64, 32, 16, 16]

        for i in range(len(out_channels)-1):
            layers.append(BasicBlock_UP(out_channels[i], out_channels[i + 1],dropout=dropout))



        finish = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 5, stride=2, padding=2,output_padding=1),
            nn.ConvTranspose1d(16, out_chanel, 5, stride=2, padding=2,output_padding=1))
        layers.append(finish)
        self.decoder = nn.Sequential(*layers)
        if freeze_dec==True:
            self.decoder.requires_grad_(False)
    def forward(self, x):
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    conv = ConvNet_ENC(1, 10)
    tt = torch.randn((1,1,1024))
    tt_ = conv.forward(tt)
    print(tt_.shape)