import torch.nn as nn


# Generator
class DCGenerator(nn.Module):
    def __init__(self, param):
        super(DCGenerator, self).__init__()
        self.param.gpu_usage = param.gpu_usage
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(param.in_channels, param.out_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(param.out_channels * 8),
            nn.ReLU(True),
            # state size. (param.out_channels*8) x 4 x 4
            nn.ConvTranspose2d(param.out_channels * 8, param.out_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.out_channels * 4),
            nn.ReLU(True),
            # state size. (param.out_channels*4) x 8 x 8
            nn.ConvTranspose2d(param.out_channels * 4, param.out_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.out_channels * 2),
            nn.ReLU(True),
            # state size. (param.out_channels*2) x 16 x 16
            nn.ConvTranspose2d(param.out_channels * 2,     param.out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.out_channels),
            nn.ReLU(True),
            # state size. (param.out_channels) x 32 x 32
            nn.ConvTranspose2d(param.out_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.param.gpu_usage > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.param.gpu_usage))
        else:
            output = self.main(input)
        return output


# Discriminator
class DCDiscriminator(nn.Module):
    def __init__(self, param):
        super(DCDiscriminator, self).__init__()
        self.param.gpu_usage = param.gpu_usage
        self.main = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, param.in_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (param.in_channels) x 32 x 32
            nn.Conv2d(param.in_channels, param.in_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (param.in_channels*2) x 16 x 16
            nn.Conv2d(param.in_channels * 2, param.in_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (param.in_channels*4) x 8 x 8
            nn.Conv2d(param.in_channels * 4, param.in_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(param.in_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (param.in_channels*8) x 4 x 4
            nn.Conv2d(param.in_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.param.gpu_usage > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.param.gpu_usage))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)