import torch
import torch.nn as nn
import numpy as np
from h_vae_model_copy import ResEncoderN
import math

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
            nn.Conv2d(self.in_width,self.middle_width,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.middle_width,self.middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.middle_width,self.middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.middle_width,self.out_width,1,1,0),
        )
        self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0)
        self.pool = nn.AvgPool2d(self.down_rate)
        self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate, mode='bilinear')

    def forward(self, x):
        xhat = self.conv(x)
        if self.residual:
            if self.in_width != self.out_width:
                x = self.size_conv(x)
            xhat = x + xhat
        if self.down_rate is not None:
            xhat = self.pool(xhat)
        if self.up_rate is not None:
            xhat = self.up_pool(xhat)
        return xhat


class TBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_width, up_rate=None, add_enc=True):
        super().__init__()
        self.up_rate = up_rate
        self.conv_pr = nn.Sequential(
            nn.Conv2d(in_width,middle_width,1,1,0),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width*3,1,1,0),
        )
        s = 2 if add_enc else 1
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_width*s,middle_width,1,1,0),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width,3,1,1),
            nn.ReLU(),
            nn.Conv2d(middle_width,middle_width*2,1,1,0),
        )
        self.conv_z = nn.Conv2d(middle_width, middle_width,1,1,0)
        self.r_block = RBlock(middle_width, middle_width, out_width, up_rate=self.up_rate, residual=True)

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def kl2(self, mu1, logvar1, mu2, logvar2):
        return (-0.5 + (logvar2/2) - (logvar1/2) + 0.5 * ((logvar1).exp() + (mu1 - mu2) ** 2) / ((logvar2).exp())).sum() / mu1.shape[0]

    # def kl2(self, mu1, logvar1, mu2, logvar2):
    #     return (-0.5 + (logvar2/2) - (logvar1/2) + 0.5 * ((logvar1).exp() + (mu1 - mu2) ** 2) / ((logvar2).exp())).mean()

    def forward(self, x, res_inp=None, train=True):
        pm, pv, pr_out = self.conv_pr(x).chunk(3, dim=1)

        x = x + pr_out
        if train:
            if res_inp is not None:
                qm, qv = self.conv_q(torch.cat([x, res_inp], dim=1)).chunk(2, dim=1)
            else:
                qm, qv = self.conv_q(x).chunk(2, dim=1)
            kl = self.kl2(qm, qv, pm, pv)
            z = self.reparametrize(qm, qv)
        else:
            z = self.reparametrize(pm, pv)
        z = self.conv_z(z)
        x = x + z
        x = self.r_block(x)
        if train:
            return x, kl
        return x



# class RBlock(nn.Module):
#     def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=True):
#         super().__init__()
#         self.down_rate = down_rate
#         self.up_rate = up_rate
#         self.residual = residual
#         self.in_width = in_width
#         self.middle_width = middle_width
#         self.out_width = out_width
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.in_width,self.middle_width,1,1,0),
#             nn.ReLU(),
#             nn.Conv2d(self.middle_width,self.middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(self.middle_width,self.middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(self.middle_width,self.out_width,1,1,0),
#             # nn.Conv2d(self.in_width,self.middle_width,3,1,1,bias=False),
#             # # nn.BatchNorm2d(self.middle_width),
#             # nn.ReLU(),
#             # nn.Conv2d(self.middle_width,self.out_width,3,1,1,bias=False),
#             # # nn.BatchNorm2d(self.out_width),
#         )
#         self.size_conv = nn.Conv2d(self.in_width, self.out_width,1,1,0)
#         self.pool = nn.AvgPool2d(self.down_rate)
#         self.up_pool = torch.nn.Upsample(scale_factor=self.up_rate, mode='bilinear')

#     def forward(self, x):
#         xhat = self.conv(x)
#         if self.residual:
#             if self.in_width != self.out_width:
#                 x = self.size_conv(x)
#             xhat = x + xhat
#         if self.down_rate is not None:
#             xhat = self.pool(xhat)
#         if self.up_rate is not None:
#             xhat = self.up_pool(xhat)
#         return xhat


# class TBlock(nn.Module):
#     def __init__(self, in_width, middle_width, out_width, up_rate=None, add_enc=True):
#         super().__init__()
#         self.up_rate = up_rate
#         self.conv_pr = nn.Sequential(
#             nn.Conv2d(in_width,middle_width,1,1,0),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width*3,1,1,0),
#             # nn.Conv2d(in_width,middle_width,3,1,1,bias=False),
#             # nn.ReLU(),
#             # nn.Conv2d(middle_width,middle_width,3,1,1,bias=False),
#             # nn.ReLU(),
#             # nn.Conv2d(middle_width,middle_width*3,3,1,1,bias=False),
#             # nn.Conv2d(in_width,middle_width,3,1,1,bias=False),
#             # nn.BatchNorm2d(middle_width),
#             # nn.GELU(),
#             # nn.Conv2d(middle_width,middle_width*3,3,1,1,bias=False),
#             # nn.BatchNorm2d(middle_width*3),
#         )
#         s = 2 if add_enc else 1
#         self.conv_q = nn.Sequential(
#             nn.Conv2d(in_width*s,middle_width,1,1,0),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(middle_width,middle_width*2,1,1,0),
#             # nn.Conv2d(in_width*s,middle_width,3,1,1,bias=False),
#             # nn.ReLU(),
#             # nn.Conv2d(middle_width,middle_width,3,1,1,bias=False),
#             # nn.ReLU(),
#             # nn.Conv2d(middle_width,middle_width*2,3,1,1,bias=False),
#             # nn.Conv2d(in_width*s,middle_width,3,1,1,bias=False),
#             # nn.BatchNorm2d(middle_width),
#             # nn.GELU(),
#             # nn.Conv2d(middle_width,middle_width*2,3,1,1,bias=False),
#             # nn.BatchNorm2d(middle_width*2),
#         )
#         self.conv_z = nn.Conv2d(middle_width, middle_width,1,1,0,bias=False)
#         self.r_block = RBlock(middle_width, middle_width, out_width, up_rate=self.up_rate, residual=True)

#     def reparametrize(self, mu, logvar):
#         noise = torch.normal(mean=0, std=1, size=mu.shape)
#         noise = noise.to(mu.device)
#         return mu + (torch.exp(logvar/2) * noise)
    
#     def kl2(self, mu1, logvar1, mu2, logvar2):
#         return (-0.5 + (logvar2/2) - (logvar1/2) + 0.5 * ((logvar1).exp() + (mu1 - mu2) ** 2) / ((logvar2).exp())).sum() / mu1.shape[0]

#     def forward(self, x, res_inp=None, train=True):
#         pm, pv, pr_out = self.conv_pr(x).chunk(3, dim=1)

#         x = x + pr_out
#         if train:
#             if res_inp is not None:
#                 qm, qv = self.conv_q(torch.cat([x, res_inp], dim=1)).chunk(2, dim=1)
#             else:
#                 qm, qv = self.conv_q(x).chunk(2, dim=1)
#             kl = self.kl2(qm, qv, pm, pv)
#             z = self.reparametrize(qm, qv)
#         else:
#             z = self.reparametrize(pm, pv)
#         z = self.conv_z(z)
#         x = x + z
#         x = self.r_block(x)
#         if train:
#             return x, kl
#         return x


class Encoder(nn.Module):
    def __init__(self, channel_list, size_in=32, size_z=64, img_ch=3):
        super().__init__()
        self.channel_list = channel_list
        self.size_z = size_z
        self.img_ch = img_ch

        self.size_in = size_in
        init_size = self.size_in
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.in_conv = nn.Conv2d(self.img_ch, self.channel_list[0][0],3,1,1)

        self.r_blocks = nn.ModuleList([RBlock(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        r_block_outs = []
        x = self.in_conv(x)
        for r_block in self.r_blocks:
            x = r_block(x)
            r_block_outs.append(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu, logvar, r_block_outs

    
class Decoder(nn.Module):
    def __init__(self, channel_list, res_inp_list, img_ch=3):
        super().__init__()
        self.channel_list = channel_list
        self.res_inp_list = res_inp_list
        self.res_inp_bool = []
        self.img_ch = img_ch
        self.out_conv = nn.Conv2d(self.channel_list[-1][2], self.img_ch,3,1,1)
        for i in range(len(channel_list)-1,-1,-1):
            if i in self.res_inp_list:
                self.res_inp_bool.append(True)
            else:
                self.res_inp_bool.append(False)
        self.t_blocks = nn.ModuleList([TBlock(*j, self.res_inp_bool[i]) for i,j in enumerate(self.channel_list)])

    def forward(self, z, enc_outs, train=True):
        kl_outs = []
        for i, t_block in enumerate(self.t_blocks):
            if train:
                if len(self.t_blocks)-1-i in self.res_inp_list:
                    z, kl = t_block(z, res_inp=enc_outs[len(self.t_blocks)-1-i])
                else:
                    z, kl = t_block(z)
                kl_outs.append(kl)
            else:
                z = t_block(z, train=False)
        z = torch.sigmoid(self.out_conv(z))
        if train:
            return z, kl_outs
        return z


class HVAE(nn.Module):

    def __init__(self, enc_channel_list, dec_channel_list, res_inp_list, size_in, size_z=64):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.res_inp_list = res_inp_list
        self.size_in = size_in
        self.size_z = size_z
        self.enc = Encoder(self.enc_channel_list, self.size_in, self.size_z)
        self.dec = Decoder(self.dec_channel_list, self.res_inp_list)
        self.learnable_feat = torch.nn.Parameter(torch.full((1,self.size_z), 0)[0].float(), requires_grad=True)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z * 2, self.size_z_lin)
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar, enc_outs = self.enc(x)
        return mu, logvar, enc_outs

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z, enc_outs, train):
        z = self.z_lin(torch.cat([z, self.learnable_feat.repeat(z.shape[0],1)], dim=1))
        if train:            
            out, kl_outs = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size), enc_outs, train)
            return out, kl_outs
        out = self.dec(z.view(z.shape[0],self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size), None, False)
        return out

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples, None, False)
    
    def forward(self, x, train=True):
        mu, logvar, enc_outs = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        if train:
            out, kl_outs = self.decoder(z, enc_outs, train)
            return out, mu, logvar, kl_outs
        else:
            out = self.decoder(z, None, False)
        return out, mu, logvar


# enc_channel_list = [(3,32,32,2), (32,64,64,2), (64,128,128,2), (128,128,128,2)]
# dec_channel_list = [(128,128,128,2), (128,128,64,2), (64,64,32,2), (32,32,3,2)]
# res_inp_list = [0,1,2,3]
# size_in = 32
# size_z=64
# pmvae0 = HVAE(enc_channel_list, dec_channel_list, res_inp_list, size_in, size_z)

class ConvCelebA(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu_lin = nn.Linear(1024, self.size_z)
        self.logvar_lin = nn.Linear(1024, self.size_z)

        self.z_lin = nn.Linear(self.size_z, 1024)
        # Mnist decoder network
        self.dec = nn.Sequential(
            nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3,kernel_size=4, stride=2),                               
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin(z).view(z.shape[0],1024,1,1)
        return self.dec(z)

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar

class SigConvCelebA(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=True)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu_lin = nn.Linear(1024, self.size_z)
        self.logvar_lin = nn.Linear(1024, self.size_z)

        self.z_lin = nn.Linear(self.size_z, 1024)
        # Mnist decoder network
        self.dec = nn.Sequential(
            nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3,kernel_size=4, stride=2),                               
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin(z).view(z.shape[0],1024,1,1)
        return self.dec(z)

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar

class ResEncoder(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64):
        super().__init__()
        self.channel_list = channel_list
        self.size_z = size_z

        self.size_in = size_in
        init_size = self.size_in
        for i in self.channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.channel_list[-1][2] // 2)

        self.r_blocks = nn.ModuleList([RBlock(*i) for i in self.channel_list])
        self.mu_lin = nn.Linear(self.size_z_lin, self.size_z)
        self.logvar_lin = nn.Linear(self.size_z_lin, self.size_z)
    
    def forward(self, x):
        for r_block in self.r_blocks:
            x = r_block(x)
        mu, logvar = x.chunk(2, dim=1)
        mu = self.mu_lin(mu.view(mu.shape[0], -1))
        logvar = self.logvar_lin(logvar.view(logvar.shape[0],-1))
        return mu, logvar

class ResDecoder(nn.Module):
    def __init__(self, channel_list, size_in=64, size_z=64):
        super().__init__()
        self.channel_list = channel_list
        self.size_z = size_z
        self.r_blocks = nn.ModuleList([RBlock(i[0],i[1],i[2],None,i[3],True) for i in self.channel_list])
        
    def forward(self, x):
        for r_block in self.r_blocks:
            x = r_block(x)
        return x

class ResCelebA(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z)
        self.dec = ResDecoder(self.dec_channel_list, self.size_in, self.size_z)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin(z)
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

class SigResCelebA(nn.Module):
    def __init__(self, enc_channel_list, dec_channel_list, size_in=64, size_z=64):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_z = size_z
        self.size_in = size_in

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z)
        self.dec = ResDecoder(self.dec_channel_list, self.size_in, self.size_z)

        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=True)

        self.size_in = size_in
        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

    def encoder(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin(z)
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



class CelebAAttr(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.enc = nn.Sequential(
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
        )
        self.mu_lin = nn.Linear(100, self.size_z)
        self.logvar_lin = nn.Linear(100, self.size_z)

        self.z_lin = nn.Linear(self.size_z, 100)
        # Mnist decoder network
        self.dec = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,40),
        )

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        z = self.z_lin(z)
        return self.dec(z)

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

class CelebAAttrNew(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.enc_net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        self.logvar_lin = nn.Linear(512, self.size_z)

        #  decoder network
        self.dec_net = nn.Sequential(
            nn.Linear(self.size_z, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,18),
        )

    def encoder(self, x):
        x = self.enc_net(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        return self.dec_net(z)

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar


class CelebAAttrNewBN(nn.Module):
    def __init__(self, size_z=64, att_size=18):
        super().__init__()
        self.att_size = att_size

        self.size_z = size_z
        self.enc_net = nn.Sequential(
            nn.Linear(self.att_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        self.logvar_lin = nn.Linear(512, self.size_z)

        #  decoder network
        self.dec_net = nn.Sequential(
            nn.Linear(self.size_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,self.att_size),
        )

    def encoder(self, x):
        x = self.enc_net(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def decoder(self, z):
        return self.dec_net(z)

    def sample(self, amount, device):
        samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        mu, logvar = self.encoder(m)
        z = self.reparametrize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
    
class CelebAAttrNewBNEnc(nn.Module):
    def __init__(self, att_size=40, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.att_size = att_size
        self.enc_net = nn.Sequential(
            nn.Linear(self.att_size, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        self.logvar_lin = nn.Linear(512, self.size_z)

    def forward(self, x):
        x = self.enc_net(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar
    
class CelAttrEnc(nn.Module):
    def __init__(self, att_size=40, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.att_size = att_size
        self.enc_net = nn.Sequential(
            nn.Linear(self.att_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        self.logvar_lin = nn.Linear(512, self.size_z)

    def forward(self, x):
        x = self.enc_net(x)
        mu, logvar = self.mu_lin(x), self.logvar_lin(x)
        return mu, logvar


class CelebAAttrNewBNAE(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.enc_net = nn.Sequential(
            nn.Linear(18, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        # self.logvar_lin = nn.Linear(512, self.size_z)

        #  decoder network
        self.dec_net = nn.Sequential(
            nn.Linear(self.size_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,18),
        )

    def encoder(self, x):
        x = self.enc_net(x)
        mu = self.mu_lin(x)
        return mu

    def decoder(self, z):
        return self.dec_net(z)

    # def sample(self, amount, device):
    #     samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        z = self.encoder(m)
        out = self.decoder(z)
        return out

class CelebAAttrNewBNAE40(nn.Module):
    def __init__(self, att_size=40, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.att_size = att_size
        self.enc_net = nn.Sequential(
            nn.Linear(self.att_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mu_lin = nn.Linear(512, self.size_z)
        # self.logvar_lin = nn.Linear(512, self.size_z)

        #  decoder network
        self.dec_net = nn.Sequential(
            nn.Linear(self.size_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,self.att_size),
        )

    def encoder(self, x):
        x = self.enc_net(x)
        mu = self.mu_lin(x)
        return mu

    def decoder(self, z):
        return self.dec_net(z)

    # def sample(self, amount, device):
    #     samples = torch.randn(amount, self.size_z).to(device)
        return self.decoder(samples)
    
    def forward(self, m):
        z = self.encoder(m)
        out = self.decoder(z)
        return out

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class CelAttrRep(nn.Module):
    def __init__(self, att_size=18, size_z=512):
        super().__init__()

        self.size_z = size_z
        self.enc = CelAttrEnc(att_size, size_z)
        self.proj = ProjectionHead(size_z, size_z)

    def forward(self, x):
        x, _ = self.enc(x)
        x = self.proj(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class CelPolyRep(nn.Module):
    def __init__(self, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.enc_channel_list1 = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
        self.size_in = 32
        self.img_ch = 3
        
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(size_z),
                nn.Linear(size_z, size_z),
                nn.GELU(),
                nn.Linear(size_z, size_z),
            )
        
        self.enc = ResEncoderN(self.enc_channel_list1, self.size_in, self.size_z, self.img_ch)
        self.proj = ProjectionHead(size_z, size_z)

    def forward(self, x, mod):
        x, _ = self.enc(x)
        x_time = self.time_mlp(mod)
        x = x + x_time
        x = self.proj(x)
        return x

class CelImgRep(nn.Module):
    def __init__(self, size_z=512):
        super().__init__()

        self.size_z = size_z
        self.enc_channel_list1 = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
        self.size_in = 128
        self.img_ch = 3
        
        self.enc = ResEncoderN(self.enc_channel_list1, self.size_in, self.size_z, self.img_ch)
        self.proj = ProjectionHead(size_z, size_z)

    def forward(self, x):
        x, _ = self.enc(x)
        x = self.proj(x)
        return x
    
class CelMaskRep(nn.Module):
    def __init__(self, size_z=512):
        super().__init__()

        self.size_z = size_z
        self.enc_channel_list1 = [(64,128,128,4), (128,256,256,4)]
        self.size_in = 128
        self.img_ch = 1
        
        self.enc = ResEncoderN(self.enc_channel_list1, self.size_in, self.size_z, self.img_ch)
        self.proj = ProjectionHead(size_z, size_z, dropout=0.2)

    def forward(self, x):
        x, _ = self.enc(x)
        x = self.proj(x)
        return x

class CtrvModel(nn.Module):
    def __init__(self, model1, model2, size_z):
        super().__init__()

        self.size_z = size_z
        self.model1 = model1
        self.proj1 = ProjectionHead(size_z, size_z)
        self.model2 = model2
        self.proj2 = ProjectionHead(size_z, size_z)
        
    def forward(self, x1, x2):
        z1, _ = self.model1(x1)
        z1 = self.proj1(z1)
        z2, _ = self.model2(x2)
        z2 = self.proj2(z2)
        return z1, z2
    
class CtrvModelGen(nn.Module):
    def __init__(self, models, size_z):
        super().__init__()

        self.size_z = size_z
        self.models = models
        
    def forward(self, xs):
        zs = []
        for i, x in enumerate(xs):
            z = self.models[i](x)
            zs.append(z)
        return zs