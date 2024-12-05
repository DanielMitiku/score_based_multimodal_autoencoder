import torch
import torch.nn as nn
from itertools import combinations
# from mnist_model import VAE
# from fashion_model import FVAE
from h_vae_model_copy import ResVAE, ResVAEN
from h_vae_model import CelebAAttrNew, CelebAAttrNewBN

class MOPOE(nn.Module):

    def __init__(self, device, vaes, size_z=10):
        super().__init__()
        
        self.size_z = size_z
        self.device = device

        self.vae_list = nn.ModuleList()
        for vae in vaes:
            self.vae_list.append(vae(device, size_z).to(device))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.vae_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            if isinstance(self.vae_list[i], FVAE):
                mu, logvar = self.vae_list[i].f_encoder(input)
            else:
                mu, logvar = self.vae_list[i].m_encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def mixture_component_selection(self, all_mus, all_logvars):
        "Taken from https://github.com/thomassutter/MoPoE/utils/utils.py"
        #if not defined, take pre-defined weights
        mus = torch.cat([mu.unsqueeze(0) for mu in all_mus], dim=0)
        logvars = torch.cat([logvar.unsqueeze(0) for logvar in all_logvars], dim=0)
        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        w_modalities = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(mus.device)
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
                i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
        return [mu_sel, logvar_sel]

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            if isinstance(self.vae_list[i], FVAE):
                mu, logvar = self.vae_list[i].f_encoder(inputs[i])
            else:
                mu, logvar = self.vae_list[i].m_encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            mu, logvar = self.poe(mus, logvars)
        else:
            mu, logvar = mus[0], logvars[0]
        z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            if isinstance(self.vae_list[i], FVAE):
                outs.append(self.vae_list[i].f_decoder(z))
            else:
                outs.append(self.vae_list[i].m_decoder(z))
        return outs

    def forward(self, inputs):
        mus, logvars = self.calc_latents(inputs)
        powerset_mus = self.powerset(mus)
        powerset_logvars = self.powerset(logvars)
        all_mus, all_logvars = [], []

        for i in range(len(powerset_mus)):
            if powerset_mus[i]:
                if len(powerset_mus[i]) == 1:
                    all_mus.append(powerset_mus[i][0]) 
                    all_logvars.append(powerset_logvars[i][0])
                else:
                    mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                    all_mus.append(mu_poe) 
                    all_logvars.append(logvar_poe) 

        sel_mu, sel_logvar = self.mixture_component_selection(all_mus, all_logvars)      
        selected_z = self.reparametrize(sel_mu, sel_logvar)
        outs = []
        for i in range(len(inputs)):
            if isinstance(self.vae_list[i], FVAE):
                outs.append(self.vae_list[i].f_decoder(selected_z))
            else:
                outs.append(self.vae_list[i].m_decoder(selected_z))
        return outs, all_mus, all_logvars


class MOPOEPoly(nn.Module):

    def __init__(self, device, vaes, size_z=64):
        super().__init__()
        
        self.size_z = size_z
        self.device = device

        self.vae_list = nn.ModuleList()
        for vae in vaes:
            self.vae_list.append(vae(device, size_z).to(device))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.vae_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs, sample_len=20):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].pm_encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(self.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].pm_decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs, sample_len=32):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].pm_encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            # mu, logvar = self.poe(mus, logvars)
            powerset_zs = []
            powerset_mus = self.powerset(mus)
            powerset_logvars = self.powerset(logvars)
            
            for i in range(len(powerset_mus)):
                if powerset_mus[i]:
                    if len(powerset_mus[i]) == 1:
                        powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    else:
                        mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                        powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
            
            uniform_div = mus[0].shape[0] // sample_len
            assert uniform_div >= 1
            selected_z = torch.zeros(powerset_zs[0].shape).to(self.device)
            unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
            # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
            for i in range(sample_len):
                if i == sample_len - 1:
                    selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
                else:
                    selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]  
        else:
            mu, logvar = mus[0], logvars[0]
            selected_z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].pm_decoder(selected_z))
        return outs

    def forward(self, inputs, sample_len=32):
        mus, logvars = self.calc_latents(inputs)
        powerset_mus = self.powerset(mus)
        powerset_logvars = self.powerset(logvars)
        powerset_zs = []
        all_mus, all_logvars = [], []
        random_indices = torch.randint(1, inputs[0].shape[0]*sample_len, size=(inputs[0].shape[0],))

        for i in range(len(powerset_mus)):
            if powerset_mus[i]:
                if len(powerset_mus[i]) == 1:
                    powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    all_mus.append(powerset_mus[i][0]) 
                    all_logvars.append(powerset_logvars[i][0])
                else:
                    mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                    powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
                    all_mus.append(mu_poe) 
                    all_logvars.append(logvar_poe) 
        
        # print('powerset: ', len(powerset_zs), powerset_zs[0].shape[0], flush=True)
        # unif_idx = torch.randint(powerset_zs[0].shape[0]*len(powerset_zs), size=(powerset_zs[0].shape[0],))   
        # powerset_zs = torch.cat(powerset_zs, dim=0).view(-1,powerset_zs[0].shape[-1])
        # selected_z = powerset_zs[unif_idx] 
        
        uniform_div = mus[0].shape[0] // sample_len
        assert uniform_div >= 1
        selected_z = torch.zeros(powerset_zs[0].shape).to(self.device)
        unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
        # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
        for i in range(sample_len):
            if i == sample_len - 1:
                selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
            else:
                selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]     
        
        # uniform_div = mus[0].shape[0] // len(powerset_zs)
        # selected_z = torch.zeros(powerset_zs[0].shape).to(self.device)
        # for i in range(len(powerset_zs)):
        #     if i == len(powerset_zs) - 1:
        #         selected_z[i*uniform_div:] = powerset_zs[i][i*uniform_div:]
        #     else:
        #         selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[i][i*uniform_div:i*uniform_div+uniform_div]
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].pm_decoder(selected_z))
        return outs, all_mus, all_logvars

    # def forward(self, inputs, sample_len=20):
    #     mus, logvars = self.calc_latents(inputs, sample_len)
    #     powerset_mus = self.powerset(mus)
    #     powerset_logvars = self.powerset(logvars)
    #     powerset_zs = []
    #     all_mus, all_logvars = [], []

    #     for i in range(len(powerset_mus)):
    #         if powerset_mus[i]:
    #             if len(powerset_mus[i]) == 1:
    #                 powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
    #                 all_mus.append(powerset_mus[i][0]) 
    #                 all_logvars.append(powerset_logvars[i][0])
    #             else:
    #                 mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
    #                 powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
    #                 all_mus.append(mu_poe) 
    #                 all_logvars.append(logvar_poe) 
        
    #     # print('powerset: ', len(powerset_zs), powerset_zs[0].shape[0], flush=True)
    #     unif_idx = torch.randint(powerset_zs[0].shape[0]*len(powerset_zs), size=(powerset_zs[0].shape[0],))   
    #     powerset_zs = torch.cat(powerset_zs, dim=0).view(-1,powerset_zs[0].shape[-1])
    #     selected_z = powerset_zs[unif_idx]      
        
    #     # uniform_div = mus[0].shape[0] // len(powerset_zs)
    #     # selected_z = torch.zeros(powerset_zs[0].shape).to(self.device)
    #     # for i in range(len(powerset_zs)):
    #     #     if i == len(powerset_zs) - 1:
    #     #         selected_z[i*uniform_div:] = powerset_zs[i][i*uniform_div:]
    #     #     else:
    #     #         selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[i][i*uniform_div:i*uniform_div+uniform_div]
    #     outs = []
    #     for i in range(len(inputs)):
    #         outs.append(self.vae_list[i].pm_decoder(selected_z))
    #     return outs, all_mus, all_logvars


class MOPOEPolyRes(nn.Module):

    def __init__(self, n_mod, enc_channel_list, dec_channel_list, size_z=64, size_in=32, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_in = size_in
        self.img_ch = img_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        for i in range(n_mod):
            self.vae_list.append(ResVAE(self.enc_channel_list, self.dec_channel_list, self.size_in, self.size_z, self.img_ch))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.vae_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs, sample_len=32, use_prod=False):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)
        
        if use_prod:
            z = self.reparametrize(*self.poe(mus, logvars))
            outs = []
            for i in range(len(self.vae_list)):
                outs.append(self.vae_list[i].decoder(z))
            return outs

        if len(present_mod) > 1:
            # mu, logvar = self.poe(mus, logvars)
            powerset_zs = []
            powerset_mus = self.powerset(mus)
            powerset_logvars = self.powerset(logvars)
            
            for i in range(len(powerset_mus)):
                if powerset_mus[i]:
                    if len(powerset_mus[i]) == 1:
                        powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    else:
                        mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                        powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
            
            if len(powerset_zs) < sample_len:
                sample_len = len(powerset_zs)

            if mus[0].shape[0] < sample_len:
                sample_len = mus[0].shape[0]

            uniform_div = mus[0].shape[0] // sample_len
            # if uniform_div < 1:
            #     uniform_div = mus[0].shape[0]
                # print("error: ", uniform_div, len(mus), mus[0].shape, sample_len, flush=True)
            assert uniform_div >= 1
            selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
            unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
            # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
            for i in range(sample_len):
                if i == sample_len - 1:
                    selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
                else:
                    selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]  
        else:
            mu, logvar = mus[0], logvars[0]
            selected_z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs

    def forward(self, inputs, sample_len=32):
        mus, logvars = self.calc_latents(inputs)
        powerset_mus = self.powerset(mus)
        powerset_logvars = self.powerset(logvars)
        powerset_zs = []
        all_mus, all_logvars = [], []
        random_indices = torch.randint(1, inputs[0].shape[0]*sample_len, size=(inputs[0].shape[0],))

        for i in range(len(powerset_mus)):
            if powerset_mus[i]:
                if len(powerset_mus[i]) == 1:
                    powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    all_mus.append(powerset_mus[i][0]) 
                    all_logvars.append(powerset_logvars[i][0])
                else:
                    mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                    powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
                    all_mus.append(mu_poe) 
                    all_logvars.append(logvar_poe) 
        
        # print('powerset: ', len(powerset_zs), powerset_zs[0].shape[0], flush=True)
        # unif_idx = torch.randint(powerset_zs[0].shape[0]*len(powerset_zs), size=(powerset_zs[0].shape[0],))   
        # powerset_zs = torch.cat(powerset_zs, dim=0).view(-1,powerset_zs[0].shape[-1])
        # selected_z = powerset_zs[unif_idx] 
        if len(powerset_zs) < sample_len:
            sample_len = len(powerset_zs)
        
        uniform_div = mus[0].shape[0] // sample_len
        assert uniform_div >= 1
        selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
        unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
        # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
        for i in range(sample_len):
            if i == sample_len - 1:
                selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
            else:
                selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]     
        
        # uniform_div = mus[0].shape[0] // len(powerset_zs)
        # selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
        # for i in range(len(powerset_zs)):
        #     if i == len(powerset_zs) - 1:
        #         selected_z[i*uniform_div:] = powerset_zs[i][i*uniform_div:]
        #     else:
        #         selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[i][i*uniform_div:i*uniform_div+uniform_div]
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs, all_mus, all_logvars

class MMVAEPolyRes(nn.Module):

    def __init__(self, n_mod, enc_channel_list, dec_channel_list, size_z=64, size_in=32, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_in = size_in
        self.img_ch = img_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        for i in range(n_mod):
            self.vae_list.append(ResVAE(self.enc_channel_list, self.dec_channel_list, self.size_in, self.size_z, self.img_ch))

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            uniform_div = mus[0].shape[0] // len(present_mod)
            assert uniform_div >= 1
            selected_z = torch.zeros(mus[0].shape).to(mus[0].device)
            for i in range(len(present_mod)):
                if i == len(present_mod) - 1:
                    selected_z[i*uniform_div:] = self.reparametrize(mus[i], logvars[i])[i*uniform_div:]
                else:
                    selected_z[i*uniform_div:i*uniform_div+uniform_div] = self.reparametrize(mus[i], logvars[i])[i*uniform_div:i*uniform_div+uniform_div]
        else:
            mu, logvar = mus[0], logvars[0]
            selected_z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs

    def forward(self, inputs):
        mus, logvars = self.calc_latents(inputs) 
    
        uniform_div = mus[0].shape[0] // len(self.vae_list)
        assert uniform_div >= 1
        selected_z = torch.zeros(mus[0].shape).to(mus[0].device)
        for i in range(len(self.vae_list)):
            if i == len(self.vae_list) - 1:
                selected_z[i*uniform_div:] = self.reparametrize(mus[i], logvars[i])[i*uniform_div:]
            else:
                selected_z[i*uniform_div:i*uniform_div+uniform_div] = self.reparametrize(mus[i], logvars[i])[i*uniform_div:i*uniform_div+uniform_div]     
        
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs, mus, logvars


class MVPolyRes(nn.Module):

    def __init__(self, n_mod, enc_channel_list, dec_channel_list, size_z=64, size_in=32, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_in = size_in
        self.img_ch = img_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        for i in range(n_mod):
            self.vae_list.append(ResVAE(self.enc_channel_list, self.dec_channel_list, self.size_in, self.size_z, self.img_ch))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        # if len(mus) == len(self.vae_list):
        mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
        logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        z = self.reparametrize(*self.poe(mus, logvars))
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def forward(self, inputs, k_len=1):
        # Append individual elbos
        elbo_terms = []
        mus, logvars = self.calc_latents(inputs)
        for i in range(len(mus)):
            elbo_terms.append([mus[i], logvars[i], self.vae_list[i].decoder(self.reparametrize(mus[i], logvars[i]))])

        # Append full elbo
        full_mu, full_logvar = self.poe(mus, logvars)
        full_z = self.reparametrize(full_mu, full_logvar)
        full_outs = []
        for i in range(len(inputs)):
            full_outs.append(self.vae_list[i].decoder(full_z))
        elbo_terms.append([full_mu, full_logvar, full_outs])

        # Append k sampled elbo
        if len(self.vae_list) > 2:
            n_k = torch.randint(2, len(self.vae_list), size=(1,)).item()
            sampled_k_idx = torch.randperm(len(self.vae_list))[:n_k]
        else:
            sampled_k_idx = torch.tensor([0,1])
        sampled_k_mu, sampled_k_logvar = self.poe([mus[idx] for idx in sampled_k_idx], [logvars[idx] for idx in sampled_k_idx])
        sampled_k_z = self.reparametrize(sampled_k_mu, sampled_k_logvar)
        sampled_k_outs = []
        for i in sampled_k_idx:
            sampled_k_outs.append(self.vae_list[i].decoder(sampled_k_z))
        elbo_terms.append([sampled_k_mu, sampled_k_logvar, sampled_k_idx, sampled_k_outs])

        return elbo_terms

class MVTPolyRes(nn.Module):

    def __init__(self, n_mod, enc_channel_list, dec_channel_list, size_z=64, size_in=32, img_ch=3):
        super().__init__()

        self.enc_channel_list = enc_channel_list
        self.dec_channel_list = dec_channel_list
        self.size_in = size_in
        self.img_ch = img_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        for i in range(n_mod):
            self.vae_list.append(ResVAE(self.enc_channel_list, self.dec_channel_list, self.size_in, self.size_z, self.img_ch))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        # if len(mus) == len(self.vae_list):
        # mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
        # logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        z = self.reparametrize(*self.poe(mus, logvars))
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def forward(self, inputs, k_len=1):
        # Append individual elbos
        elbo_terms = []
        mus, logvars = self.calc_latents(inputs)
        for i in range(len(mus)):
            elbo_terms.append([mus[i], logvars[i], self.vae_list[i].decoder(self.reparametrize(mus[i], logvars[i]))])

        # Append full elbo
        full_mu, full_logvar = self.poe(mus, logvars)
        full_z = self.reparametrize(full_mu, full_logvar)
        full_outs = []
        for i in range(len(inputs)):
            full_outs.append(self.vae_list[i].decoder(full_z))
        elbo_terms.append([full_mu, full_logvar, full_outs])

        return elbo_terms


#MMVAEPLUS


# ******* CELEBHQ MODELS *************
class MOPOECeleb(nn.Module):

    def __init__(self, enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=256, size_in=128, img_ch=3, mask_ch=1):
        super().__init__()

        self.enc_channel_list1 = enc_channel_list1
        self.dec_channel_list1 = dec_channel_list1
        self.enc_channel_list2 = enc_channel_list2
        self.dec_channel_list2 = dec_channel_list2
        self.size_in = size_in
        self.img_ch = img_ch
        self.mask_ch = mask_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        self.vae_list.append(ResVAEN(self.enc_channel_list1, self.dec_channel_list1, self.size_in, self.size_z, self.img_ch))
        self.vae_list.append(ResVAEN(self.enc_channel_list2, self.dec_channel_list2, self.size_in, self.size_z, self.mask_ch))
        self.vae_list.append(CelebAAttrNewBN(size_z))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.vae_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs, sample_len=32):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            # mu, logvar = self.poe(mus, logvars)
            powerset_zs = []
            powerset_mus = self.powerset(mus)
            powerset_logvars = self.powerset(logvars)
            
            for i in range(len(powerset_mus)):
                if powerset_mus[i]:
                    if len(powerset_mus[i]) == 1:
                        powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    else:
                        mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                        powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
            
            if len(powerset_zs) < sample_len:
                sample_len = len(powerset_zs)

            if mus[0].shape[0] < sample_len:
                sample_len = mus[0].shape[0]

            uniform_div = mus[0].shape[0] // sample_len
            assert uniform_div >= 1
            selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
            unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
            # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
            for i in range(sample_len):
                if i == sample_len - 1:
                    selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
                else:
                    selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]  
        else:
            mu, logvar = mus[0], logvars[0]
            selected_z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs

    def forward(self, inputs, sample_len=32):
        mus, logvars = self.calc_latents(inputs)
        powerset_mus = self.powerset(mus)
        powerset_logvars = self.powerset(logvars)
        powerset_zs = []
        all_mus, all_logvars = [], []

        for i in range(len(powerset_mus)):
            if powerset_mus[i]:
                if len(powerset_mus[i]) == 1:
                    powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    all_mus.append(powerset_mus[i][0]) 
                    all_logvars.append(powerset_logvars[i][0])
                else:
                    mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                    powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
                    all_mus.append(mu_poe) 
                    all_logvars.append(logvar_poe) 
        
        if len(powerset_zs) < sample_len:
            sample_len = len(powerset_zs)
        
        uniform_div = mus[0].shape[0] // sample_len
        assert uniform_div >= 1
        selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
        unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
        for i in range(sample_len):
            if i == sample_len - 1:
                selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
            else:
                selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]     
        
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs, all_mus, all_logvars


class MOPOECeleb2mod(nn.Module):

    def __init__(self, enc_channel_list1, dec_channel_list1, size_z=256, size_in=128, img_ch=3):
        super().__init__()

        self.enc_channel_list1 = enc_channel_list1
        self.dec_channel_list1 = dec_channel_list1
        self.size_in = size_in
        self.img_ch = img_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        self.vae_list.append(ResVAEN(self.enc_channel_list1, self.dec_channel_list1, self.size_in, self.size_z, self.img_ch))
        self.vae_list.append(CelebAAttrNewBN(size_z))

    def powerset(self, input_list):
        result = []
        n = len(input_list)
        for i in range(0,n+1):
            for element in combinations(input_list,i):
                result.append(element)
        return result

    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        if len(mus) == len(self.vae_list):
            mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
            logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs, sample_len=32):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        if len(present_mod) > 1:
            # mu, logvar = self.poe(mus, logvars)
            powerset_zs = []
            powerset_mus = self.powerset(mus)
            powerset_logvars = self.powerset(logvars)
            
            for i in range(len(powerset_mus)):
                if powerset_mus[i]:
                    if len(powerset_mus[i]) == 1:
                        powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    else:
                        mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                        powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
            
            if len(powerset_zs) < sample_len:
                sample_len = len(powerset_zs)

            if mus[0].shape[0] < sample_len:
                sample_len = mus[0].shape[0]

            uniform_div = mus[0].shape[0] // sample_len
            assert uniform_div >= 1
            selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
            unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
            # unif_idx = torch.randint(len(powerset_zs), size=(sample_len,))
            for i in range(sample_len):
                if i == sample_len - 1:
                    selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
                else:
                    selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]  
        else:
            mu, logvar = mus[0], logvars[0]
            selected_z = self.reparametrize(mu, logvar)

        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs

    def forward(self, inputs, sample_len=32):
        mus, logvars = self.calc_latents(inputs)
        powerset_mus = self.powerset(mus)
        powerset_logvars = self.powerset(logvars)
        powerset_zs = []
        all_mus, all_logvars = [], []

        for i in range(len(powerset_mus)):
            if powerset_mus[i]:
                if len(powerset_mus[i]) == 1:
                    powerset_zs.append(self.reparametrize(powerset_mus[i][0], powerset_logvars[i][0]))
                    all_mus.append(powerset_mus[i][0]) 
                    all_logvars.append(powerset_logvars[i][0])
                else:
                    mu_poe, logvar_poe = self.poe(powerset_mus[i], powerset_logvars[i])
                    powerset_zs.append(self.reparametrize(mu_poe, logvar_poe))
                    all_mus.append(mu_poe) 
                    all_logvars.append(logvar_poe) 
        
        if len(powerset_zs) < sample_len:
            sample_len = len(powerset_zs)
        
        uniform_div = mus[0].shape[0] // sample_len
        assert uniform_div >= 1
        selected_z = torch.zeros(powerset_zs[0].shape).to(mus[0].device)
        unif_idx = torch.randperm(len(powerset_zs))[:sample_len]
        for i in range(sample_len):
            if i == sample_len - 1:
                selected_z[i*uniform_div:] = powerset_zs[unif_idx[i]][i*uniform_div:]
            else:
                selected_z[i*uniform_div:i*uniform_div+uniform_div] = powerset_zs[unif_idx[i]][i*uniform_div:i*uniform_div+uniform_div]     
        
        outs = []
        for i in range(len(inputs)):
            outs.append(self.vae_list[i].decoder(selected_z))
        return outs, all_mus, all_logvars
    

class MVTCeleb(nn.Module):

    def __init__(self, enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=256, size_in=128, img_ch=3, mask_ch=1):
        super().__init__()

        self.enc_channel_list1 = enc_channel_list1
        self.dec_channel_list1 = dec_channel_list1
        self.enc_channel_list2 = enc_channel_list2
        self.dec_channel_list2 = dec_channel_list2
        self.size_in = size_in
        self.img_ch = img_ch
        self.mask_ch = mask_ch
        self.size_z = size_z

        self.vae_list = nn.ModuleList()
        self.vae_list.append(ResVAEN(self.enc_channel_list1, self.dec_channel_list1, self.size_in, self.size_z, self.img_ch))
        self.vae_list.append(ResVAEN(self.enc_channel_list2, self.dec_channel_list2, self.size_in, self.size_z, self.mask_ch))
        self.vae_list.append(CelebAAttrNewBN(size_z))
    
    def poe(self, mus, logvars):
        mu = torch.cat([mu.unsqueeze(0) for mu in mus], dim=0)
        logvar = torch.cat([logvar.unsqueeze(0) for logvar in logvars], dim=0)
        # if len(mus) == len(self.vae_list):
        # mu = torch.cat([mu, torch.zeros(1,mu.shape[1],mu.shape[2]).to(mu.device)], dim=0)
        # logvar = torch.cat([logvar, torch.zeros(1,logvar.shape[1],logvar.shape[2]).to(mu.device)], dim=0)
        var = torch.exp(logvar) + 1e-8
        T = 1 / var
        poe_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0)
        poe_var = 1 / torch.sum(T, dim=0)
        return poe_mu, torch.log(poe_var + 1e-8)

    def calc_latents(self, inputs):
        mus, logvars = [], []
        for i, input in enumerate(inputs):
            mu, logvar = self.vae_list[i].encoder(input)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def reparametrize(self, mu, logvar):
        noise = torch.normal(mean=0, std=1, size=mu.shape)
        noise = noise.to(mu.device)
        return mu + (torch.exp(logvar/2) * noise)

    def sample(self, z):
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs

    def cond_gen(self, present_mod, inputs):
        mus, logvars = [], []
        for i in present_mod:
            mu, logvar = self.vae_list[i].encoder(inputs[i])
            mus.append(mu)
            logvars.append(logvar)

        z = self.reparametrize(*self.poe(mus, logvars))
        outs = []
        for i in range(len(self.vae_list)):
            outs.append(self.vae_list[i].decoder(z))
        return outs
    
    def forward(self, inputs, k_len=1):
        # Append individual elbos
        elbo_terms = []
        mus, logvars = self.calc_latents(inputs)
        for i in range(len(mus)):
            elbo_terms.append([mus[i], logvars[i], self.vae_list[i].decoder(self.reparametrize(mus[i], logvars[i]))])

        # Append full elbo
        full_mu, full_logvar = self.poe(mus, logvars)
        full_z = self.reparametrize(full_mu, full_logvar)
        full_outs = []
        for i in range(len(inputs)):
            full_outs.append(self.vae_list[i].decoder(full_z))
        elbo_terms.append([full_mu, full_logvar, full_outs])

        return elbo_terms