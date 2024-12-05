import torch
import torch.nn as nn
from itertools import combinations
# from mnist_model import VAE
# from fashion_model import FVAE
from h_vae_model_copy import ResDecoder, ResEncoder, ResVAE, ResEncoderN, ResDecoderN
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import torch.distributions as dist
from numpy import sqrt

#MMVAEPLUS

# Taken and sUpdated from https://github.com/epalu/mmvaeplus/

def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean

class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes, mod_types=None):
        super(MMVAE, self).__init__()
        self.pu = prior_dist
        self.pw = prior_dist
        if mod_types is None:
            self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        else:
            self.vaes = nn.ModuleList([vaes[i](params, mod_types[i]) for i in range(len(vaes))])
            print('mods: ', mod_types, flush=True)
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pu_params = None  # defined in subclass

    @property
    def pu_params(self):
        return self._pu_params

    # def getDataLoaders(batch_size, shuffle=True, device="cuda"):
    #     # handle merging individual datasets appropriately in sub-class
    #     raise NotImplementedError

    def forward(self, x, K=1):
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if self.params.variant == 'mmvaeplus':
                        _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                        pw = self.pw(*vae.pw_params)
                        latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                        if not self.params.no_cuda and torch.cuda.is_available():
                            latents_w.cuda()
                        us_combined = torch.cat((latents_w, z_e), dim=-1)
                        px_us[e][d] = vae.px_u(*vae.dec(us_combined))
                    # elif self.params.variant == 'mmvaefactorized':
                    #     _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    #     us_target = uss[d]
                    #     w_d, _ = torch.split(us_target, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    #     us_combined = torch.cat((w_d, z_e), dim=-1)
                    #     px_us[e][d] = vae.px_u(*vae.dec(us_combined))
                    else:
                        raise ValueError("wrong option for variant paramter")
        return qu_xs, px_us, uss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pu = self.pu(*self.pu_params)
            latents = pu.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def cond_gen(self, present_mod, inputs):
        #crossmodalmatrix inference
        recons = self.reconstruct_and_cross_reconstruct(inputs)
        
        if len(present_mod) == 1:
            result =  recons[present_mod[0]]
        else:
            random_idx = torch.randint(0, len(present_mod), (1,)).item()
            selected_mod = present_mod[random_idx]
            result = recons[selected_mod]
        # remove the added 1 dimension at the 0 dim and return
        return [r[0] for r in result]

    def reconstruct_and_cross_reconstruct_forw(self, data):
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # pw = self.pz(torch.zeros(1, self.params.latent_dim_w), torch.ones(1, self.params.latent_dim_w))
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=1)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            pu = self.pu(*self.pu_params)
            latents_u_to_split = pu.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
            latents_w_new, _ = torch.split(latents_u_to_split, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            us = torch.cat((latents_w_new, latents_z), dim=-1)
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.px_u(*vae.dec(us))
        return qu_xs, px_us, uss

    def reconstruct_and_cross_reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_us, _ = self.reconstruct_and_cross_reconstruct_forw(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pu = prior_dist
        self.px_u = likelihood_dist
        self.qu_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pu_params = None  # defined in subclass
        self._qu_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

        self._pw_params = None # defined in subclass

    @property
    def pu_params(self):
        return self._pu_params

    @property
    def pw_params(self):
        return self._pw_params

    @property
    def qu_x_params(self):
        if self._qu_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qu_x_params

    # @staticmethod
    # def getDataLoaders(batch_size, shuffle=True, device="cuda"):
    #     # handle merging individual datasets appropriately in sub-class
    #     raise NotImplementedError

    def forward(self, x, K=1):
        self._qu_x_params = self.enc(x)
        qu_x = self.qu_x(*self._qu_x_params)
        us = qu_x.rsample(torch.Size([K]))
        px_u = self.px_u(*self.dec(us))
        return qu_x, px_u, us

    def generate(self, N, K):   # Not exposed as here we only train multimodal VAES
        self.eval()
        with torch.no_grad():
            pu = self.pu(*self.pu_params)
            latents = pu.rsample(torch.Size([N]))
            px_u = self.px_u(*self.dec(latents))
            data = px_u.sample(torch.Size([K]))
        return data.view(-1, *data.size()[3:])

    def reconstruct(self, data):  # Not exposed as here we only train multimodal VAES
        self.eval()
        with torch.no_grad():
            qu_x = self.qu_x(*self.enc(data))
            latents = qu_x.rsample()  # no dim expansion
            px_u = self.px_u(*self.dec(latents))
            recon = get_mean(px_u)
        return recon

# Encoder network
class Enc(nn.Module):
    def __init__(self, ndim_w=32, ndim_z=32):
        super().__init__()
        self.enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
        self.size_in = 32
        self.img_ch = 3

        self.ndim_w = ndim_w
        self.ndim_z = ndim_z
        self.size_z = self.ndim_w + self.ndim_z

        self.enc = ResEncoder(self.enc_channel_list, self.size_in, self.size_z, self.img_ch)

    def forward(self, x):
        mean, logvar= self.enc(x)
        lv_w = logvar[:,:self.ndim_w]
        lv_z = logvar[:,self.ndim_w:]

        # print("ENCODER out size: ", mean.shape, lv_w.shape, (F.softmax(lv_w, dim=-1) * lv_w.size(-1) + 1e-6).shape)

        return mean, \
               torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + 1e-6,
                          F.softmax(lv_z, dim=-1) * lv_z.size(-1) + 1e-6), dim=-1)

# Decoder network
class Dec(nn.Module):
    def __init__(self, ndim=64):
        super().__init__()

        self.enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
        self.dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
        self.size_in = 32
        self.img_ch = 3
        self.size_z = ndim

        self.dec_ = ResDecoder(self.dec_channel_list, self.size_in, self.size_z, self.img_ch)

        init_size = self.size_in
        for i in self.enc_channel_list:
            init_size = init_size // i[3]
        self.size_z_lin = (init_size * init_size) * (self.enc_channel_list[-1][2])

        self.z_lin = nn.Linear(self.size_z, self.size_z_lin)
        self.z_lin_relu = nn.ReLU()
        self.z_reshape_size = (self.size_z_lin // self.enc_channel_list[-1][2] // init_size)

        
    def forward(self, u):
        z = u.view(u.shape[0]*u.shape[1],*u.shape[2:])
        # print("DEcoder input shapae: ", z.shape)
        z = self.z_lin_relu(self.z_lin(z))
        # print("z2 shape: ", z.shape)
        out = self.dec_(z.view(z.shape[0], self.enc_channel_list[-1][2],self.z_reshape_size,self.z_reshape_size))
        out = out.view(*u.size()[:2], *out.size()[1:])
        return out, torch.tensor(0.75).to(u.device)



class PolyMNIST(VAE):
    def __init__(self, params):
        super(PolyMNIST, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            # dist.Normal,  # prior
            # dist.Normal,  # likelihood
            # dist.Normal,  # posterior
            Enc(params.latent_dim_w, params.latent_dim_z),
            Dec(params.latent_dim_w + params.latent_dim_z),
            params
        )
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False)  # logvar
        ])
        grad_w = {'requires_grad': params.learn_prior_w_polymnist}
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), **grad_w)  # logvar
        ])
        self.modelName = 'polymnist_resnet'
        # self.dataSize = dataSize
        self.llik_scaling = 1.
        self.params = params

    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)

    @property
    def pw_params(self):
        return self._pw_params[0], F.softmax(self._pw_params[1], dim=1) * self._pw_params[1].size(-1)
    '''
    @property
    def pu_params(self):
        return self._pu_params[0], F.softplus(self._pu_params[1]) + 1e-6
    @property
    def pw_params(self):
        return self._pw_params[0], F.softplus(self._pw_params[1]) + 1e-6
    '''

    # def getDataLoaders(self, batch_size, shuffle=True, device='cuda', m=0):
    #     unim_train_datapaths = [self.tmpdir + "/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
    #     unim_test_datapaths = [self.tmpdir + "/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
    #     kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    #     tx = transforms.ToTensor()
    #     train = DataLoader(PolyMNISTDataset(unim_train_datapaths, transform=tx),
    #                        batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     test = DataLoader(PolyMNISTDataset(unim_test_datapaths, transform=tx),
    #                        batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     return train, test

    def generate(self, runPath, epoch): # NOT EXPOSED: we only train multimodal VAEs here
        N, K = 64, 9
        samples = super(PolyMNIST, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):  # NOT EXPOSED: we only train multimodal VAEs here
        recon = super(PolyMNIST, self).reconstruct(data)
        comp = torch.cat([data, recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

class PolyMNIST_10modalities(MMVAE):
    def __init__(self, params):
        n_mod = params.n_mod
        vae_list = [PolyMNIST for _ in range(n_mod)]
        super(PolyMNIST_10modalities, self).__init__(dist.Laplace, params, *vae_list)
        # PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, \
        # PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST )
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False)  # logvar
        ])
        # REMOVE LLIK SCALING
        # self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            # if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'polymnist-10modalities'
        # Fix model names for indiviudal models to be saved
        for idx, vae in enumerate(self.vaes):
            vae.modelName = 'polymnist_m'+str(idx)
            vae.llik_scaling = 1.0
        self.tmpdir = params.tmpdir

    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)

    #@property
    # def pu_params(self):
    #    return self._pu_params[0], F.softplus(self._pu_params[1]) + Constants.eta

    #def setTmpDir(self, tmpdir):
    #    self.tmpdir = tmpdir

    # def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
    #     tx = transforms.ToTensor()
    #     unim_train_datapaths = [self.tmpdir+"/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
    #     unim_test_datapaths = [self.tmpdir+"/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
    #     dataset_PolyMNIST_train = PolyMNISTDataset(unim_train_datapaths, transform=tx)
    #     dataset_PolyMNIST_test = PolyMNISTDataset(unim_test_datapaths, transform=tx)
    #     kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    #     train = DataLoader(dataset_PolyMNIST_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     test = DataLoader(dataset_PolyMNIST_test, batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     return train, test

    def generate(self):
        N = 100
        outputs = []
        samples_list = super(PolyMNIST_10modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            samples = samples.view(N, *samples.size()[1:])
            outputs.append(make_grid(samples, nrow=int(sqrt(N))))
        return outputs
    
    def unc_gen(self, N):
        return super(PolyMNIST_10modalities, self).generate(N)

    def generate_for_calculating_unconditional_coherence(self, N):
        samples_list = super(PolyMNIST_10modalities, self).generate(N)
        return [samples.data.cpu() for samples in samples_list]

    def generate_for_fid(self, savedir, num_samples, tranche):
        N = num_samples
        samples_list = super(PolyMNIST_10modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            for image in range(samples.size(0)):
                save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savedir, i, tranche, image))

    def reconstruct_for_fid(self, data, savedir, i):
        recons_mat = super(PolyMNIST_10modalities, self).reconstruct_and_cross_reconstruct([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                recon = recon.squeeze(0).cpu()
                for image in range(recon.size(0)):
                    save_image(recon[image, :, :, :],
                                '{}/m{}/m{}/{}_{}.png'.format(savedir, r,o, image, i))

    def cross_generate(self, data):
        N = 10
        recon_triess = [[[] for i in range(N)] for j in range(N)]
        outputss = [[[] for i in range(N)] for j in range(N)]
        for i in range(10):
            recons_mat = super(PolyMNIST_10modalities, self).reconstruct_and_cross_reconstruct([d[:N] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                      recon = recon.squeeze(0).cpu()
                      recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([data[r][:N].cpu()]+recon_triess[r][o]), nrow=N)
        return outputss

