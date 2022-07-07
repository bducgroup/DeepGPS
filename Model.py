import torch
import torch.nn as nn
import functools
import numpy as np


class KingsleyModel(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # Encoder
        self.model = ResnetEncoder(3, 1, n_blocks=4).to(self.device)
        # Position Decoder
        self.model2 = ResnetDecoder(1, 1, n_blocks=6).to(self.device)
        # TimeMLP
        self.model3 = TimeEffector().to(self.device)

    def forward(self, matrix, time, skyplot):
        # three types of input:
        # environment, timestamp, skyplot 
        matrix = matrix.to(self.device)
        time = time.to(self.device)
        skyplot = skyplot.to(self.device)

        # generate constraint mask
        constraint = matrix > 0
        constraint_mask = torch.ones(matrix.shape)
        constraint_mask[constraint] = 0

        # generate timestamp matrix
        time2matrix = self.model3(time).unsqueeze(1)

        # combine to be input matrices 
        main_input = torch.cat([matrix, time2matrix], dim=1)
        main_input = torch.cat([main_input, skyplot], dim=1)

        # feed input_matrices into Encoder
        self.Encoder_output = self.model(main_input)
        
        # Output corrected_position
        output = self.model2(self.Encoder_output)
        
        #Process with constraint mask
        result = output * constraint_mask.to(self.device)

        return result


class DistanceDecoder(nn.Module):
    def __init__(self):
        super(DistanceDecoder, self).__init__()
        self.cov = [
            nn.Conv2d(256, 512, kernel_size=3, stride=3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        ]
        self.mlp = [
            nn.Linear(4096, 2048), nn.ReLU(True),
            nn.Linear(2048, 1024), nn.ReLU(True),
            nn.Linear(1024, 512), nn.ReLU(True),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, 25), nn.ReLU(True),
        ]
        self.cov = nn.Sequential(*self.cov)
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, time):
        output = self.cov(time)
        # print('output shape',output.shape)
        output = output.view(len(output), -1)
        output = self.mlp(output)
        # output = output.view(len(time),100,100)
        return output


class TimeEffector(nn.Module):
    def __init__(self):
        super(TimeEffector, self).__init__()
        self.model = [
            nn.Linear(7, 100), nn.ReLU(True),
            nn.Linear(100, 1000), nn.ReLU(True),
            nn.Linear(1000, 2000), nn.ReLU(True),
            nn.Linear(2000, 10000), nn.ReLU(True)
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, time):
        output = self.model(time)
        output = output.view(len(time), 100, 100)
        return output

# define encoder
class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='zero'):
        assert (n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.Cov1 = [nn.Conv2d(input_nc, input_nc, kernel_size=7, padding=3,
                               bias=use_bias),
                     norm_layer(input_nc),
                     nn.Tanh()]
        self.Cov1 = nn.Sequential(*self.Cov1)

        self.Cov2 = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                               bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]
        self.Cov2 = nn.Sequential(*self.Cov2)

        self.Cov3 = []
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            self.Cov3 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        self.Cov3 = nn.Sequential(*self.Cov3)

        mult = 2 ** n_downsampling
        self.Encoder_Res = []
        for i in range(n_blocks):
            self.Encoder_Res += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.Encoder_Res = nn.Sequential(*self.Encoder_Res)

    def forward(self, input):
        out = self.Cov1(input)
        out = self.Cov2(out)
        out = self.Cov3(out)
        encoder_output = self.Encoder_Res(out)
        return encoder_output

# define decoder
class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='zero', encoder_blocks=6):
        assert (n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.Decoder_Res = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            if i <= encoder_blocks:
                continue
            self.Decoder_Res += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.Decoder_Res = nn.Sequential(*self.Decoder_Res)

        self.CovT1 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.CovT1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=0,
                                              bias=use_bias),
                           norm_layer(int(ngf * mult / 2)),
                           nn.ReLU(True)]
        self.CovT1 = nn.Sequential(*self.CovT1)
        self.Cov4 = []
        self.Cov4 += [nn.Conv2d(ngf, output_nc, kernel_size=4, padding=[3, 3])]
        self.Cov4 += [nn.Sigmoid()]

        self.Cov4 = nn.Sequential(*self.Cov4)

    def forward(self, input):
        out = self.Decoder_Res(input)
        out = self.CovT1(out)
        out = self.Cov4(out)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.25)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def get_accuracy(predictions, labels):
    predictions = predictions.squeeze().cpu().detach().numpy()
    labels = labels.squeeze().cpu().detach().numpy()
    predictions = np.array([np.unravel_index(i.argmax(), predictions.shape) for i in predictions])
    labels = np.array([np.unravel_index(i.argmax(), labels.shape) for i in labels])

    delta = predictions - labels
    delta = delta ** 2
    delta_sum1 = delta.sum(1)

    max_of_delta = max(delta_sum1)
    sum_of_delta = np.sqrt(delta_sum1).mean()
    del predictions, labels
    return sum_of_delta, max_of_delta
