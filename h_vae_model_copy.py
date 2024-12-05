import torch
import torch.nn as nn
import numpy as np
import torchvision
import math
import torch.nn.functional as F


class RBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.sf = nn.LeakyReLU(0.2)
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate)

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat

class ResEncoder(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlock(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu, logvar

class ResDecoder(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.r_blocks = nn.ModuleList([RBlock(i[0],i[1],i[2],None,i[3],True) for i in self.channel_list])
        self.ch_dec = nn.Sequential(
            RBlock(self.channel_list[-1][2], self.channel_list[-1][2], self.channel_list[-1][2]),
            nn.Conv2d(self.channel_list[-1][2], self.img_ch, 5, 1, 2)
        )

    def forward(self, x):
        for r_block in self.r_blocks:
            x = r_block(x)
        x = self.ch_dec(x)
        return x

class ResVAE(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)
        self.dec = ResDecoder(self.dec_channel_list, self.size_in, self.size_z, self.img_ch)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar

class ResAE(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)
        self.dec = ResDecoder(self.dec_channel_list, self.size_in, self.size_z, self.img_ch)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, _ = self.enc(x)
        return mu

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out
    
    def forward(self, m):
        mu = self.encoder(m)
        out = self.decoder(mu)
        return out

class RBlock2(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.ReLU(),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.sf = nn.ReLU()
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate)

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat

class ResCLF(nn.Module):
    def __init__(self, channel_list, size_in=64, size_out=18, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.size_out = size_out
        self.channel_list = channel_list
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.ReLU(),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_clf_lin = (init_size * init_size) * (self.channel_list[-1][2])

        self.r_blocks = nn.ModuleList([RBlock2(*i) for i in self.channel_list])
        self.clf_lin = nn.Linear(self.size_clf_lin, self.size_out)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)
        out = self.clf_lin(x.view(x.shape[0], -1))
        return out

class Res50CLF(nn.Module):
    def __init__(self, size_out=18):
        super().__init__()
        self.size_out = size_out
        self.res50 = torchvision.models.resnet50(pretrained=True)
        res_modules = list(self.res50.children())[:-1]
        self.res50 = nn.Sequential(*res_modules)
        for p in self.res50.parameters():
            p.requires_grad = False
        self.res50.eval()
        self.clf_net = nn.Linear(2048, self.size_out)
    
    def forward(self, x):
        x = self.res50(x).view(-1,2048)
        return self.clf_net(x)


class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=1, b=1):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.a*residual + self.b*out;
        return out

def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample,a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, a, b):
        super(FeatureExtractorImg, self).__init__();
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(3, 128,
                              kernel_size=3,
                              stride=2,
                              padding=2,
                              dilation=1,
                              bias=False)
        self.resblock1 = make_res_block_feature_extractor(128, 2 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock2 = make_res_block_feature_extractor(2 * 128, 3 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock3 = make_res_block_feature_extractor(3 * 128, 4 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock4 = make_res_block_feature_extractor(4 * 128, 5 * 128, kernelsize=4, stride=2,
                                                          padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out

class ClfImg(nn.Module):
    def __init__(self):
        super(ClfImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg(a=2.0, b=0.3);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=5*128, out_features=18, bias=True);
        self.sigmoid = nn.Sigmoid();

    def forward(self, x_img):
        h = self.feature_extractor(x_img);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        return h;

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img);
        return h;


# New ResVAE

class RBlockN(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.GELU(),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.sf = nn.GELU()
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate, mode='bilinear')

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat

class ResEncoderN(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlockN(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu, logvar

class ResDecoderN(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.r_blocks = nn.ModuleList([RBlockN(i[0],i[1],i[2],None,i[3],True) for i in self.channel_list])
        self.ch_dec = nn.Sequential(
            RBlock(self.channel_list[-1][2], self.channel_list[-1][2], self.channel_list[-1][2]),
            nn.Conv2d(self.channel_list[-1][2], self.img_ch, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        for r_block in self.r_blocks:
            x = r_block(x)
        x = self.ch_dec(x)
        return x

class ResDecoderSoft(nn.Module):
    def __init__(self, channel_list, last_enc_ch, init_size, size_z_lin, size_in=64, size_z=64, img_ch=3):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.r_blocks = nn.ModuleList([RBlock(i[0],i[1],i[2],None,i[3],True) for i in self.channel_list])
        self.ch_dec = nn.Sequential(
            RBlock(self.channel_list[-1][2], self.channel_list[-1][2], self.channel_list[-1][2]),
            nn.Conv2d(self.channel_list[-1][2], self.img_ch, 5, 1, 2),
            nn.Sigmoid()
        )
        self.last_enc_ch = last_enc_ch
        self.size_z_lin = size_z_lin
        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.last_enc_ch // init_size)

    def forward(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        z = z.view(z.shape[0],self.last_enc_ch,self.z_reshape_size,self.z_reshape_size)

        for r_block in self.r_blocks:
            z = r_block(z)
        x = self.ch_dec(z)
        return x

class ResVAEN(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch

        self.enc = ResEncoderN(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)
        self.dec = ResDecoderN(self.dec_channel_list, self.size_in, self.size_z, self.img_ch)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar

class ResVAESoft(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)
        self.dec = ResDecoderSoft(self.dec_channel_list,self.enc_channel_list[-1][2], init_size, self.size_z_lin, self.size_in, self.size_z, self.img_ch)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        out = self.dec(z)
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar


class ResAEN(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch

        self.enc = ResEncoderN(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)
        self.dec = ResDecoderN(self.dec_channel_list, self.size_in, self.size_z, self.img_ch)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, _ = self.enc(x)
        return mu

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out

    # def sample(self, amount, device):
    #     samples = torch.randn(amount, self.size_z).to(device)
    #     return self.decoder(samples)
    
    def forward(self, m):
        z = self.encoder(m)
        out = self.decoder(z)
        return out


# Res VAE with dropout

class RBlockND(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True, drop_p=0.25):
        super().__init__()
        self.down_rate = down_rate
        self.drop_p = drop_p
        self.up_rate = up_rate
        self.residual = residual
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.middle_width),
            nn.GELU(),
            nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_width),
            nn.Dropout(p=self.drop_p),
        )
        self.sf = nn.GELU()
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0,bias=False)
        self.down_pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate, mode='bilinear')

    def forward(self, x):
        xhat = self.conv(x)
        if self.in_width != self.out_width:
            x = self.size_conv(x)
        xhat = self.sf(x + xhat)
        if self.down_rate is not None:
            xhat = self.down_pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat

class ResEncoderND(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3, drop_p=0.25):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.drop_p = drop_p
        self.ch_enc = nn.Sequential(
            nn.Conv2d(self.img_ch, self.channel_list[0][0], 5, 1, 2),
            nn.BatchNorm2d(self.channel_list[0][0]),
            nn.GELU(),
            nn.Dropout(p=self.drop_p),
            nn.AvgPool2d(2),
        ) 

        self.size_in = size_in
        init_size = self.size_in // 2
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlockND(*i,drop_p=self.drop_p) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        x = self.ch_enc(x)
        for r_block in self.r_blocks:
            x = r_block(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu, logvar

class ResDecoderND(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64, img_ch=3, drop_p=0.25):
        super().__init__()
        self.img_ch = img_ch
        self.channel_list = channel_list
        self.size_z = size_z
        self.drop_p = drop_p
        self.r_blocks = nn.ModuleList([RBlockND(i[0],i[1],i[2],None,i[3],True,drop_p=self.drop_p) for i in self.channel_list])
        self.ch_dec = nn.Sequential(
            RBlockND(self.channel_list[-1][2], self.channel_list[-1][2], self.channel_list[-1][2]),
            nn.Conv2d(self.channel_list[-1][2], self.img_ch, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        for r_block in self.r_blocks:
            x = r_block(x)
        x = self.ch_dec(x)
        return x

class ResAEND(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3, drop_p=0.25):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch
        self.drop_p = drop_p

        self.enc = ResEncoderND(self.enc_channel_list, self.size_in, self.size_z, self.img_ch, self.drop_p)
        self.dec = ResDecoderND(self.dec_channel_list, self.size_in, self.size_z, self.img_ch, self.drop_p)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, _ = self.enc(x)
        return mu

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out

    # def sample(self, amount, device):
    #     samples = torch.randn(amount, self.size_z).to(device)
    #     return self.decoder(samples)
    
    def forward(self, m):
        z = self.encoder(m)
        out = self.decoder(z)
        return out
    
class ResVAEND(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64, img_ch=3, drop_p=0.25):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in
        self.img_ch = img_ch
        self.drop_p = drop_p

        self.enc = ResEncoderND(self.enc_channel_list, self.size_in, self.size_z, self.img_ch, self.drop_p)
        self.dec = ResDecoderND(self.dec_channel_list, self.size_in, self.size_z, self.img_ch, self.drop_p)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin_relu(self.z_lin(z))
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar
    
############ RESVAEMMPLUS #############
# Model taken and updated from MMPLUS for replication

# Constants
dataSize = torch.Size([3, 28, 28])

# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Encoder network
class Enc(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim_w=32, ndim_z=32):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_z = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)

    def forward(self, x):
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
        lv_w = self.fc_lv_w(out_w)

        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size()[0], self.nf0 * self.s0 * self.s0)
        lv_z = self.fc_lv_z(out_z)

        mu, logvar = torch.cat((self.fc_mu_w(out_w), self.fc_mu_z(out_z)), dim=-1), \
               torch.cat((lv_w, lv_z), dim=-1)
        return mu, logvar
    
class Enc2(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim_z=64):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        # self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        # self.resnet_w = nn.Sequential(*blocks_w)
        # self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        # self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_z = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)

    def forward(self, x):
        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size()[0], self.nf0 * self.s0 * self.s0)
        lv_z = self.fc_lv_z(out_z)

        mu, logvar = self.fc_mu_z(out_z), lv_z
        return mu, logvar

# Decoder network
class Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, ndim=64):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, u):
        out = self.fc(u).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        # out = out.view(*u.size()[:2], *out.size()[1:])
        # consider also predicting the length scale
        return out #, torch.tensor(0.75).to(u.device)  # mean, length scale
    
class EncN(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim_w=32, ndim_z=32):
        super().__init__()
        s0 = self.s0 = 8  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 128

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_z = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0*s0*s0, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0*s0*s0, ndim_z)

    def forward(self, x):
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
        lv_w = self.fc_lv_w(out_w)

        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size()[0], self.nf0*self.s0*self.s0)
        lv_z = self.fc_lv_z(out_z)

        return torch.cat((self.fc_mu_w(out_w), self.fc_mu_z(out_z)), dim=-1), \
               torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta,
                          F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta), dim=-1)

# Decoder network
class DecN(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, ndim=64):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 8  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 128

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            # if i == nlayers - 1:
            #     nf1 = 128
            # print('nf: ', nf0, nf1)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, u):
        out = self.fc(u).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = out.view(*u.size()[:2], *out.size()[1:])
        # consider also predicting the length scale
        return out

class ResVAEMMPLUS(nn.Module):
    def __init__(self, size_z=64):
        # size_z must be even int
        super().__init__()

        self.size_z = size_z

        self.enc = Enc(self.size_z//2, self.size_z//2)
        self.dec = Dec(self.size_z)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        out = self.dec(z)
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar
    
class ResVAEMMPLUSN(nn.Module):
    def __init__(self, size_z=64):
        # size_z must be even int
        super().__init__()

        self.size_z = size_z

        self.enc = EncN(self.size_z//2, self.size_z//2)
        self.dec = DecN(self.size_z)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        out = self.dec(z)
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar).unsqueeze(1)
        out = self.decoder(z)

        return out, mu, logvar
    
class ResVAEMMPLUS2(nn.Module):
    def __init__(self, size_z=64):
        # size_z must be even int
        super().__init__()

        self.size_z = size_z

        self.enc = Enc2(self.size_z)
        self.dec = Dec(self.size_z)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        out = self.dec(z)
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar