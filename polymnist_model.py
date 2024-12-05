import torch
import torch.nn as nn
import numpy as np

class NView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class PMVAE(nn.Module):

    def __init__(self, device, size_z=512):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),                                             
            nn.Linear(2048, 512),
            nn.ReLU(),
            )
        self.pm_mu = nn.Linear(512, self.size_z)
        self.pm_logvar = nn.Linear(512, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 2048),
            View((-1,128,4,4)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), 
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar


class PMVAE2(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),                                             
            ))
        self.pm_mu = nn.Linear(1024, self.size_z)
        self.pm_logvar = nn.Linear(1024, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 256*2*2),
            View((-1,256,2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=0), 
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar

class PMVAE3(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),                                             
            )
        self.pm_mu = nn.Linear(4096, self.size_z)
        self.pm_logvar = nn.Linear(4096, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 1024*2*2),
            View((-1,1024,2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=0), 
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar

class PMAE64(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),
            nn.Linear(1024, self.size_z)                                             
            ))

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(64, 256*2*2),
            View((-1,256,2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=0), 
        )

    def pm_encoder(self, x):
        return self.pm_encoder_net(x)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        z = self.pm_encoder(m)
        pm_out = self.pm_decoder(z)

        return pm_out, z

class SigPMVAE(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=True)

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),
            )
        self.pm_mu = nn.Linear(1024, self.size_z)
        self.pm_logvar = nn.Linear(1024, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 256*2*2),
            View((-1,256,2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=0), 
            nn.Sigmoid()
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar

class SigVAE(nn.Module):

    def __init__(self, device, size_z=128):
        super().__init__()

        self.size_z = size_z
        self.device = device
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=True)

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4,512),
            nn.ReLU(),
        )
        self.pm_mu = nn.Linear(512, self.size_z)
        self.pm_logvar = nn.Linear(512, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 256*4*4),
            View((-1,256,4,4)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=0),                 
            nn.Sigmoid(),
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar


class PMCLF(nn.Module):

    def __init__(self):
        super().__init__()

        # clf network
        self.clf_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            NView(),                                             
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            )

    def forward(self, x):
        return self.clf_net(x)

class PTRCLF(nn.Module):

    def __init__(self):
        super().__init__()

        # clf network
        self.clf_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.25), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.25),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            NView(),                                             
            nn.Linear(2048, 512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 10),
            )

    def forward(self, x):
        return self.clf_net(x)


class P2MOPOE(nn.Module):

    def __init__(self, device, num_modality, size_z=64):
        super().__init__()
        
        self.size_z = size_z
        self.device = device

        self.vae_list = nn.ModuleList()
        for i in range(num_modality):
            self.vae_list.append(PMVAE2(device, size_z))

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / (var + 1e-8)
        poe_mu = torch.sum(mu*T, dim=0) * (1 / torch.sum(T, dim=0))
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].pm_encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize_all(self, mus, logvars):
        zs = []
        for i in range(len(mus)):
            mu, logvar = mus[i], logvars[i]
            noise = torch.normal(mean=0, std=1, size=mu.shape)
            noise = noise.to(self.device)
            zs.append(mu + (torch.exp(logvar/2) * noise))
        return zs

    def reparametrize_single(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def forward(self, inputs):
        mus, logvars = self.calc_latents(inputs)
        zs = self.reparametrize_all(mus, logvars)
        mu_poe, logvar_poe = self.poe(mus, logvars)
        poe_z = self.reparametrize_single(mu_poe, logvar_poe)
        mus, logvars = mus + [mu_poe], logvars + [logvar_poe]
        zs = zs + [poe_z]
        sampled_mixture = np.random.choice(3, 1).item()
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].pm_decoder(zs[sampled_mixture]))
        return outs, mus, logvars

class Unflatten(nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)

class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__()

        self.flags = flags
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # content branch
        self.class_mu = nn.Linear(1024, flags.class_dim)
        self.class_logvar = nn.Linear(1024, flags.class_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        return None, None, self.class_mu(h), self.class_logvar(h)


class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.decoder = nn.Sequential(
            nn.Linear(flags.style_dim + flags.class_dim, 256*2*2),
            Unflatten((256, 2, 2)),    
            nn.ReLU(),                                                      
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=0), 
        )

    def forward(self, style_latent_space, class_latent_space):
        z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(z.device)

class InfMoPoE(nn.Module):
    """
    This is a class built for inference for PolyMnist models trained using the official github code.
    """
    def __init__(self, enc_dec_list, device, size_z=64):
        super().__init__()
        self.size_z=64
        self.device = device
        self.enc_dec_list = enc_dec_list

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.enc_dec_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(self.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(self.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / (var + 1e-8)
        poe_mu = torch.sum(mu*T, dim=0) * (1 / torch.sum(T, dim=0))
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            _, _, mu, logvar = self.enc_dec_list[i][0](inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            mu, logvar = self.poe(mus, logvars)
        else:
            mu, logvar = mus[0], logvars[0]
        z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.enc_dec_list)):
            outs.append(self.enc_dec_list[i][1](None,z)[0])
        return outs
    
    def sample(self, z):
        outs = []
        for i in range(len(self.enc_dec_list)):
            outs.append(self.enc_dec_list[i][1](None,z)[0])
        return outs


class InfExp(nn.Module):
    """
    This is a class built for inference for PolyMnist models trained using the official github code.
    """
    def __init__(self, enc_dec_list, device, size_z=64):
        super().__init__()
        self.size_z=64
        self.device = device
        self.enc_dec_list = enc_dec_list

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.enc_dec_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(self.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(self.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / (var + 1e-8)
        poe_mu = torch.sum(mu*T, dim=0) * (1 / torch.sum(T, dim=0))
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def mixture_component_selection(self, all_mus, all_logvars):
        "Taken from https://github.com/thomassutter/MoPoE/utils/utils.py"
        #if not defined, take pre-defined weights
        mus = torch.cat([mu.unsqueeze(0) for mu in all_mus], dim=0)
        logvars = torch.cat([logvar.unsqueeze(0) for logvar in all_logvars], dim=0)
        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        w_modalities = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == w_modalities.shape[0]-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        return [mu_sel, logvar_sel]

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def cond_gen(self, present_mod, inputs, type_exp='moe'):
        mus, logvars = [], []
        for i in present_mod:
            _, _, mu, logvar = self.enc_dec_list[i][0](inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if type_exp == 'poe':
            if len(present_mod) > 1:
                mu, logvar = self.poe(mus, logvars)
            else:
                mu, logvar = mus[0], logvars[0]

        elif type_exp == 'moe':
            if len(present_mod) > 1:
                mu, logvar = self.mixture_component_selection(mus, logvars)
            else:
                mu, logvar = mus[0], logvars[0]

        z = self.reparametrize(mu, logvar)
        outs = []
        for i in range(len(self.enc_dec_list)):
            outs.append(self.enc_dec_list[i][1](None,z)[0])
        return outs
    
    def sample(self, z):
        outs = []
        for i in range(len(self.enc_dec_list)):
            outs.append(self.enc_dec_list[i][1](None,z)[0])
        return outs


class ConvPoly(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            NView(),                                             
            )

        self.pm_mu = nn.Linear(1024, self.size_z)
        self.pm_logvar = nn.Linear(1024, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 256*2*2),
            View((-1,256,2,2)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0),   
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),                 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=0), 
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar


class ConvPoly2(nn.Module):

    def __init__(self, device, size_z=64):
        super().__init__()

        self.size_z = size_z
        self.device = device

        # Mnist encoder network
        self.pm_encoder_net = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            NView(),                                             
            )

        self.pm_mu = nn.Linear(1024, self.size_z)
        self.pm_logvar = nn.Linear(1024, self.size_z)

        # Mnist decoder network
        self.pm_decoder_net = nn.Sequential(
            nn.Linear(self.size_z, 256*2*2),
            View((-1,256,2,2)),
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=0),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),                 
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=5, stride=1, padding=0), 
        )

    def pm_encoder(self, x):
        x = self.pm_encoder_net(x)
        mu, logvar = self.pm_mu(x), self.pm_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def pm_decoder(self, z):
        return self.pm_decoder_net(z)
    
    def forward(self, m):
        mu, logvar = self.pm_encoder(m)
        z = self.reparametrize(mu, logvar)
        pm_out = self.pm_decoder(z)

        return pm_out, mu, logvar