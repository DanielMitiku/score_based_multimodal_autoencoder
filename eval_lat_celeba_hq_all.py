import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import glob
from PIL import Image
import math
from inspect import isfunction
from functools import partial
from einops import rearrange
from torch import nn, einsum

from celeba_hq_mask_dataset import CelebAHQMaskDS

from h_vae_model import ResCelebA, CelebAAttrNewBN, CelebAAttrNewBNAE
from h_vae_model_copy import ResVAEN, ResAEN
from mopoe_model import MOPOECeleb, MVTCeleb
from mmplus_model_cel import MMPLUSCeleba
from unet_model import Unet, CAUNET
from unet_openai import UNetModel
from utils import *
from sklearn.metrics import f1_score
from pytorch_fid.fid_score import calculate_fid_given_paths


def get_test_dataloader(batch_size, size):
    test_dataset = CelebAHQMaskDS(size=size, ds_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataloader

def unimodal_fid(test_loader, univae, device, path):
    with torch.no_grad():
        univae.eval()
        start_time = time.time()
        
        for batch_idx, (images, _, _) in enumerate(test_loader):

            images = images.to(device)
            # masks = masks.to(device)
            # target = target.to(device)[:, attr_visible]

            save_batch_image(images, path['in_image'] + str(batch_idx) + '_')
            save_batch_image(univae.sample(images.shape[0], device), path['out_image_univae'] + str(batch_idx) + '_')

        print('calculating FID UniVAE')
        fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_univae']], 256, device, 2048, 2)
        
        print("Image FID Univae: ", fid_img)
        end_time = time.time()

        print("Uni VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        return

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def evaluate_diff_vae(vae_out, sm_model, device):
    with torch.no_grad():
        print("Difvae started", flush=True)
        sm_model.eval()
        diff_outs = []
        start_time = time.time()

        timesteps = 1000
        betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # diff_outs.extend([sample_in, vae_out])
        sample = torch.normal(mean=0, std=1, size=vae_out.shape, device=device)
        vae_out = (vae_out * 2) -1

        for t in range(timesteps-1, -1, -1):

            t_batch = torch.full((sample.shape[0],), t, device=device, dtype=torch.long)
            betas_t = extract(betas, t_batch, sample.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                sqrt_one_minus_alphas_cumprod, t_batch, sample.shape
            )
            sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_batch, sample.shape)
            
            model_mean_t = sqrt_recip_alphas_t * (
                sample - betas_t * sm_model(torch.cat([sample, vae_out],dim=1), t_batch) / sqrt_one_minus_alphas_cumprod_t
            )

            if t > 0:
                posterior_variance_t = extract(posterior_variance, t_batch, sample.shape)
                noise = torch.randn_like(sample)
                sample = model_mean_t + torch.sqrt(posterior_variance_t) * noise
            else:
                sample = model_mean_t
            # print('time ', t, flush=True)
        
        print("Difvae VALIDATION TIME TAKEN: ", time.time() - start_time, flush=True)
        return (sample + 1) / 2


def evaluate_mopoe(test_loader, mix_vae, mix_type, attr_visible, att_threshold, size_z, device, path, given):
    with torch.no_grad():
        mix_vae.eval()
        start_time = time.time()
        f1_att = f1_mask = 0
        correct_att, total_att = 0, 0
        correct_mask, total_mask = 0, 0
        true_att, predicted_att = [], []
        true_mask, predicted_mask = [], []
        mods = '012'

        present_mod_list = []
        for mod in mods:
            if mod in given:
                present_mod_list.append(int(mod))

        for batch_idx, (images, masks, target) in enumerate(test_loader):

            images = images.to(device)
            masks = masks.to(device)
            target = target.to(device)[:, attr_visible]

            samples = {'0': images, '1': masks, '2': target.float()}
            present = []
            for mod in mods:
                if mix_type == 'mmplus':
                    present.append(samples[mod])
                else:
                    if mod in given:
                        present.append(samples[mod])
                    else:
                        present.append(None)
                

            if len(given) >= 1:
                all_predicted = mix_vae.cond_gen(present_mod_list, present)
            else:
                if mix_type == 'mmplus':
                    all_predicted = mix_vae.unc_gen(images.shape[0])
                else:
                    z = torch.normal(mean=0, std=1, size=(images.shape[0],size_z), device=device)
                    all_predicted = mix_vae.sample(z)

            # Calc att F1
            if mix_type == 'mmplus':
                sigmoid_outputs = all_predicted[2].cpu()
            else:
                sigmoid_outputs = torch.sigmoid(all_predicted[2]).cpu()
            # predicted_att_round = np.round(sigmoid_outputs)
            predicted_att_round = sigmoid_outputs > att_threshold

            true_att.append(target.cpu())
            predicted_att.append(predicted_att_round)
            total_att += target.shape[0] * target.shape[1]
            correct_att += (predicted_att_round == target.cpu()).sum().item()

            # Calc mask F1
            mask_outputs = all_predicted[1].detach().cpu()
            predicted_mask_round = np.round(mask_outputs)

            input_mask_round = np.round(masks.cpu())
            true_mask.append(input_mask_round.view(masks.shape[0],-1))
            predicted_mask.append(predicted_mask_round.view(masks.shape[0],-1))
            total_mask += torch.prod(torch.tensor(masks.shape))
            correct_mask += (predicted_mask_round == input_mask_round.cpu()).sum().item()
            
            # Calc Fid Image
            save_batch_image(images, path['in_image'] + str(batch_idx) + '_')
            save_batch_image(all_predicted[0], path['out_image_' + mix_type] + str(batch_idx) + '_')
            
        print('calculating FID ', mix_type)
        fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_' + mix_type]], 256, device, 2048, 2)
        
        print("Image FID ", mix_type, ': ', fid_img)
        f1_mask = f1_score(torch.cat(true_mask, dim=0).numpy(), torch.cat(predicted_mask, dim=0).numpy(), average='samples')
        f1_att = f1_score(torch.cat(true_att, dim=0).numpy(), torch.cat(predicted_att, dim=0).numpy(), average='samples')
        end_time = time.time()

        print(mix_type, " VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print(mix_type, " F1 Attribute: ", f1_att, flush=True)
        print(mix_type, " Acc Attribute: ", correct_att/total_att)
        print(mix_type, " F1 Mask: ", f1_mask, flush=True)
        print(mix_type, " Acc Mask: ", correct_mask/total_mask)
        return
            

@torch.no_grad()
def evaluate(val_loader, image_vae, mask_vae, attr_vae, attr_visible, att_threshold, n_mod, model, difvae, device, size_z, er, n_comp, given, path, c, is_vae=True):
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    model.eval()
    difvae.eval()
    # if is_vae:
    #     sigmas = torch.tensor(np.linspace(3, 0.1, 200)).to(device)
    # else:
    #     sigmas = torch.tensor(np.linspace(10, 0.01, 500)).to(device)
    sigmas = torch.tensor(np.linspace(5, 0.1, 500)).to(device)
    start_time = time.time()
    dim = int(np.sqrt(size_z)) # pass size_z that is a perfect square or change this

    f1_att = f1_mask = 0
    correct_att, total_att = 0, 0
    correct_mask, total_mask = 0, 0
    true_att, predicted_att = [], []
    true_mask, predicted_mask = [], []

    for batch_idx, (images, masks, target) in enumerate(val_loader):

        images = images.to(device)
        masks = masks.to(device)
        target = target.to(device)[:, attr_visible]

        mods = '012' # 0 for image, 1 for mask, 2 for attr
        z = {}
        models = {'0': image_vae, '1': mask_vae, '2': attr_vae}
        samples = {'0': images, '1': masks, '2': target.float()}
        mod_to_word = {'0': 'IMAGE', '1': 'MASK', '2': 'TARGET'}
        outs = {}

        given_string = "GIVEN "

        for mod in mods:
            if mod in given:
                if is_vae:
                    z[mod] = models[mod].encoder(samples[mod])[0]
                else:
                    z[mod] = models[mod].encoder(samples[mod])
                given_string += mod_to_word[mod] + " "
            else:
                z[mod] = torch.normal(mean=0, std=1, size=(images.shape[0], size_z), device=device)
            outs[mod] = models[mod].decoder(z[mod])
    
        for s_in, s in enumerate(sigmas):
            sigma_index = torch.tensor([s_in]*images.shape[0]).to(device)
            cur_sigmas = sigmas[sigma_index].float().to(device)
            # alpha = er * (sigmas[s_in]**2)/(sigmas[-1]**2)

            # noise = {}
            # for mod in given:
            #     noise[mod] = s * torch.randn_like(z[mod])
            #     z[mod] = z[mod] + noise[mod]

            for i in range(n_comp):
                z_all = torch.cat([z[mod].unsqueeze(1) for mod in sorted(models.keys())], dim=1).view(-1,n_mod,dim,dim)
                sm_out = model(z_all, sigma_index) / cur_sigmas.view(z_all.shape[0],*([1]*len(z_all.shape[1:])))

                for ind, mod in enumerate(mods):
                    if mod not in given:
                        alpha = er[mod] * (sigmas[s_in]**2)/(sigmas[-1]**2)
                        z[mod] = z[mod] + (alpha * sm_out[:,ind].view(-1,size_z)) + c[mod]*(torch.sqrt(2*alpha) * torch.randn_like(z[mod]))

            # for mod in given:
            #     z[mod] = z[mod] - noise[mod]

        for ind, mod in enumerate(mods):
            if mod not in given:
                outs[mod] = models[mod].decoder(z[mod])
        
        # Calc att F1
        sigmoid_outputs = torch.sigmoid(outs['2']).detach().cpu()
        # predicted_att_round = np.round(sigmoid_outputs)
        predicted_att_round = sigmoid_outputs > att_threshold

        true_att.append(target.cpu())
        predicted_att.append(predicted_att_round)
        total_att += target.shape[0] * target.shape[1]
        correct_att += (predicted_att_round == target.cpu()).sum().item()

        # Calc mask F1
        mask_outputs = outs['1'].detach().cpu()
        predicted_mask_round = np.round(mask_outputs)

        input_mask_round = np.round(masks.cpu())
        true_mask.append(input_mask_round.view(masks.shape[0],-1))
        predicted_mask.append(predicted_mask_round.view(masks.shape[0],-1))
        total_mask += torch.prod(torch.tensor(masks.shape))
        correct_mask += (predicted_mask_round == input_mask_round.cpu()).sum().item()

        # Calc Fid Image
        save_batch_image(images, path['in_image'] + str(batch_idx) + '_')
        if is_vae:
            save_batch_image(outs['0'], path['out_image_vae'] + str(batch_idx) + '_')
            # if len(given) == 0:
            #     difvae_out = evaluate_diff_vae(outs['0'], difvae, device)
            #     save_batch_image(difvae_out, path['out_image_vae_difvae'] + str(batch_idx) + '_')
        else:
            save_batch_image(outs['0'], path['out_image_ae'] + str(batch_idx) + '_')
            # difvae_out = evaluate_diff_vae(outs['0'], difvae, device)
            # save_batch_image(difvae_out, path['out_image_ae_difvae'] + str(batch_idx) + '_')

        # print('done one batch!', flush=True)

    print(given_string, flush=True)
    print('calculating FID')
    if is_vae:
        fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_vae']], 256, device, 2048, 2)
        # if len(given) == 0:
        #     fid_img_difvae = calculate_fid_given_paths([path['in_image'], path['out_image_vae_difvae']], 256, device, 2048, 2)
    else:
        fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_ae']], 256, device, 2048, 2)
        # fid_img_difvae = calculate_fid_given_paths([path['in_image'], path['out_image_ae_difvae']], 256, device, 2048, 2)

    print("Image FID: ", fid_img)
    # print("Image FID Difvae: ", fid_img_difvae)
    
    f1_mask = f1_score(torch.cat(true_mask, dim=0).numpy(), torch.cat(predicted_mask, dim=0).numpy(), average='samples')
    f1_att = f1_score(torch.cat(true_att, dim=0).numpy(), torch.cat(predicted_att, dim=0).numpy(), average='samples')
    end_time = time.time()

    print("SM-UNET VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
    
    print("SM-UNET F1 Attribute score: ", f1_att, flush=True)
    print("SM-UNET Acc Attribute: ", correct_att/total_att)

    print("SM-UNET F1 Mask score: ", f1_mask, flush=True)
    print("SM-UNET Acc Mask: ", correct_mask/total_mask)

    return


def run(batch_size, size_z1, size_z2, image_vae_path, mask_vae_path, attr_vae_path, image_ae_path, mask_ae_path, attr_ae_path, score_vae_path, score_ae_path, score_openai_path, unet_vae_path,mopoe_model_path, mvt_model_path, mmplus_model_path):
    print('_______________________________________________________')
    print('vars: ', batch_size, size_z1, size_z2, flush=True)
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    print(attr_visible, flush=True)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:0")

    res_size = 128
    
    # Load mask model
    enc_channel_list = [(64,128,128,4), (128,256,256,4)]
    dec_channel_list = [(256,256,128,4), (128,128,64,4)]
    size_in = res_size
    img_ch = 1    
    mask_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    mask_vae.load_state_dict(torch.load(mask_vae_path, map_location=device)['model_state_dict'])
    mask_vae = mask_vae.to(device)

    mask_ae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    mask_ae.load_state_dict(torch.load(mask_ae_path, map_location=device)['model_state_dict'])
    mask_ae = mask_ae.to(device)

    # Load image model
    #sm
    enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
    dec_channel_list = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
    # enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2), (512,1024,1024,2)]
    # dec_channel_list = [(1024,1024,512,2), (512,512,256,2), (256,256,128,2), (128,128,64,2)]
    size_in = res_size
    img_ch = 3    
    image_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    image_vae.load_state_dict(torch.load(image_vae_path, map_location=device)['model_state_dict'])
    image_vae = image_vae.to(device)

    image_ae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    image_ae.load_state_dict(torch.load(image_ae_path, map_location=device)['model_state_dict'])
    image_ae = image_ae.to(device)

    # Load attr model
    attr_vae = CelebAAttrNewBN(size_z2)
    attr_vae.load_state_dict(torch.load(attr_vae_path, map_location=device)['model_state_dict'])
    attr_vae = attr_vae.to(device)

    attr_ae = CelebAAttrNewBNAE(size_z2)
    attr_ae.load_state_dict(torch.load(attr_ae_path, map_location=device)['model_state_dict'])
    attr_ae = attr_ae.to(device)

    assert size_z1 == size_z2

    # MoPoE VAE
    enc_channel_list1 = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
    dec_channel_list1 = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
    enc_channel_list2 = [(64,128,128,4), (128,256,256,4)]
    dec_channel_list2 = [(256,256,128,4), (128,128,64,4)]
    size_in = 128
    img_ch = 3
    
    mopoe_vae = MOPOECeleb(enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=size_z1, size_in=size_in, img_ch=img_ch, mask_ch=1)
    mopoe_vae.load_state_dict(torch.load(mopoe_model_path, map_location=device)['model_state_dict'])
    mopoe_vae = mopoe_vae.to(device)

    mvt_vae = MVTCeleb(enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=size_z1, size_in=size_in, img_ch=img_ch, mask_ch=1)
    mvt_vae.load_state_dict(torch.load(mvt_model_path, map_location=device)['model_state_dict'])
    mvt_vae = mvt_vae.to(device)

    b_size = batch_size
    class Params():
        latent_dim_w = size_z1 // 2
        latent_dim_z = size_z1 // 2
        model = 'celebhq'
        obj = 'dreg'
        K = 1
        batch_size = b_size
        epochs = 300
        beta = 1
        learn_prior_w_polymnist = True
        variant = 'mmvaeplus'
        tmpdir = '/tmp/'
        no_cuda = False
        n_mod = 3
    params = Params()

    mmplus_vae = MMPLUSCeleba(params)
    mmplus_vae.load_state_dict(torch.load(mmplus_model_path, map_location=device)['model_state_dict'])
    mmplus_vae = mmplus_vae.to(device)

    n_mod = 3
    score_model = Unet(dim=128, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True)
    score_model.load_state_dict(torch.load(score_vae_path, map_location=device)['model_state_dict'])
    score_model = score_model.to(device)

    score_model_ae = Unet(dim=128, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True)
    score_model_ae.load_state_dict(torch.load(score_ae_path, map_location=device)['model_state_dict'])
    score_model_ae = score_model_ae.to(device)

    # score_openai = UNetModel(in_channels=n_mod, model_channels=128, out_channels=n_mod, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, channel_mult=(1,2,2,2,2), num_heads=8)
    # score_openai.load_state_dict(torch.load(score_openai_path, map_location=device)['model_state_dict'])
    # score_openai = score_openai.to(device)

    unetvae = UNetModel(in_channels=img_ch*2, model_channels=128, out_channels=img_ch, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, channel_mult=(1,2,2,3,4), num_heads=8)
    unetvae.load_state_dict(torch.load(unet_vae_path, map_location=device)['model_state_dict'])
    unetvae = unetvae.to(device)

    print('Models loaded', flush=True)
    
    test_dataloader = get_test_dataloader(batch_size, res_size)

    print('Dataloader done', flush=True)

    
    n_comp = 2
    n_comp_ae = 2

    givens = ['', '0', '1', '2', '01', '02', '12']
    # # givens = ['1', '2', '12']
    # # givens = ['12']
    # givens = ['01']

    c_vae = {'0': 0.8, '1': 0.5, '2': 0.5}
    c_ae = {'0': 0.8, '1': 0.5, '2': 0.5}
    att_threshold = 0.5

    # er = {'0': 0.8, '1': 0.5, '2': 0.5}
    # er_ae = {'0': 0.8, '1': 0.5, '2': 0.5}
    rand_num = str(int(torch.rand(1)*10000))
    path = {'in_image': '/tmp/temp_hq_in' + rand_num + '/', 
            'out_image_univae': '/tmp/temp_hq_out_univae' + rand_num + '/', 
            'out_image_univae_difvae': '/tmp/temp_hq_out_univae_difvae' + rand_num + '/', 
            'out_image_vae': '/tmp/temp_hq_out_vae' + rand_num + '/', 
            'out_image_vae_difvae': '/tmp/temp_hq_out_vae_difvae' + rand_num + '/', 
            'out_image_ae': '/tmp/temp_hq_out_ae' + rand_num + '/', 
            'out_image_ae_difvae': '/tmp/temp_hq_out_ae_difvae' + rand_num + '/', 
            'out_image_mopoe': '/tmp/temp_hq_out_mopoe' + rand_num + '/', 
            'out_image_mopoe_difvae': '/tmp/temp_hq_out_mopoe_difvae' + rand_num + '/',
            'out_image_mvt': '/tmp/temp_hq_out_mvt' + rand_num + '/', 
            'out_image_mvt_difvae': '/tmp/temp_hq_out_mvt_difvae' + rand_num + '/',
            'out_image_mmplus': '/tmp/temp_hq_out_mmplus' + rand_num + '/', 
            'out_image_mmplus_difvae': '/tmp/temp_hq_out_mmplus_difvae' + rand_num + '/'}

    for p in path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    # unimodal_fid(test_dataloader, image_vae, device, path)
    # for given in ['1','2']:
    #     print('VAE with difvae') # VAE
    #     evaluate(test_dataloader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, score_model, unetvae, device, size_z1, er, n_comp, given, path, c_vae, True)
    # return 

    for given in givens:
        if len(given) == 0:
            c_vae = {'0': 0.8, '1': 0.8, '2': 0.8}
            c_ae = {'0': 0.8, '1': 0.8, '2': 0.8}
            er = {'0': 0.01, '1': 0.01, '2': 0.01}
            er_ae = {'0': 0.01, '1': 0.01, '2': 0.01}
        elif len(given) == 1:
            c_vae = {'0': 0.8, '1': 0.5, '2': 0.5}
            c_ae = {'0': 0.8, '1': 0.5, '2': 0.5}
            er = {'0': 0.016, '1': 0.015, '2': 0.015}
            er_ae = {'0': 0.016, '1': 0.015, '2': 0.015}
        elif len(given) == 2:
            c_ae = {'0': 0.8, '1': 0.5, '2': 0.5}
            er = {'0': 0.018, '1': 0.015, '2': 0.015}
            er_ae = {'0': 0.02, '1': 0.015, '2': 0.015}
            c_vae = {'0': 0.8, '1': 0.5, '2': 0}

        
        print(' ')
        print('ncomp: ', n_comp, n_comp_ae, ' er: ', er, er_ae, 'c: ', c_vae, c_ae, 'given: ', given)

        print('AE') # AE
        evaluate(test_dataloader, image_ae, mask_ae, attr_ae, attr_visible, att_threshold, n_mod, score_model_ae, unetvae, device, size_z1, er_ae, n_comp_ae, given, path, c_ae, False) 
        print('VAE') # VAE
        evaluate(test_dataloader, image_vae, mask_vae, attr_vae, attr_visible, att_threshold, n_mod, score_model, unetvae, device, size_z1, er, n_comp, given, path, c_vae, True)
        print('MoPoE') # MoPoE
        evaluate_mopoe(test_dataloader, mopoe_vae, 'mopoe', attr_visible, att_threshold, size_z1, device, path, given)
        print('MVT') # MVTCAE
        evaluate_mopoe(test_dataloader, mvt_vae, 'mvt', attr_visible, att_threshold, size_z1, device, path, given)
        print('MMPLUS') # MMVAE+
        evaluate_mopoe(test_dataloader, mmplus_vae, 'mmplus', attr_visible, att_threshold, size_z1, device, path, given)
        
        print(' ')

        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z1', type=int, default=256,
                        help='size of z1 [default: 256]')
    parser.add_argument('--size-z2', type=int, default=256,
                        help='size of z2 [default: 256]')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size for training [default: 512]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate [default: 0.0001]')
    parser.add_argument('--unq-name', type=str, default='unet_dsm',
                        help='name to identify the model [default: "unet_dsm"]')
    parser.add_argument('--vae-type', type=str, default='conv_sig',
                        help='vae type [default: "conv_sig"]')
    
    parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__',
                        help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__"]')
    parser.add_argument('--mask-path', type=str, default='./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq',
                        help='mask vae model path [default: "./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq"]')
    parser.add_argument('--attr-path', type=str, default='./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1',
                        help='vae model path [default: "./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1"]')
    
    # parser.add_argument('--unet-path', type=str, default='./models/celeb_score/celeb_hq_unetVAE_256_uhqVAE_b01_s3e2_18_uhqVAE_b01_s3e2_',
    #                     help='score path  [default: "./models/celeb_score/celeb_hq_unetVAE_256_uhqVAE_b01_s3e2_18_uhqVAE_b01_s3e2_"]')
    parser.add_argument('--unet-path', type=str, default='./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEmeanb01_z256_dim128_12222_s5_01_18_uhqVAEmeanb01_z256_dim128_12222_s5_01_',
                        help='score path  [default: "./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEmeanb01_z256_dim128_12222_s5_01_18_uhqVAEmeanb01_z256_dim128_12222_s5_01_"]')

    parser.add_argument('--image-path-ae', type=str, default='./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_',
                        help='image path for vae [default: "./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_"]')
    parser.add_argument('--mask-path-ae', type=str, default='./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_',
                        help='mask path for vae [default: "./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_"]')
    parser.add_argument('--attr-path-ae', type=str, default='./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1',
                        help='attr path for vae [default: "./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1"]')

    # parser.add_argument('--unet-path-ae', type=str, default='./models/celeb_score/celeb_hq_unet_AE_256unet_hq256_allru__18unet_hq256_allru__',
    #                     help='score path  [default: "./models/celeb_score/celeb_hq_unet_AE_256unet_hq256_allru__18unet_hq256_allru__"]')
    parser.add_argument('--unet-path-ae', type=str, default='./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_12222_s5_01_18_uhqAEnew_z256_dim128_12222_s5_01_',
                        help='score path  [default: "./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_12222_s5_01_18_uhqAEnew_z256_dim128_12222_s5_01_"]')
    parser.add_argument('--unet-openai-path', type=str, default='./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_18_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_',
                        help='openai score path  [default: "./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_18_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_"]')
    
    # parser.add_argument('--mopoe-path', type=str, default='./models/mopoe_pupd/celebhqN_mopoe_vae_res_beta_1.0__2560.0001',
    #                     help='mopoe model path [default: "./models/mopoe_pupd/celebhqN_mopoe_vae_res_beta_1.0__2560.0001"]')
    parser.add_argument('--mopoe-path', type=str, default='./models/mopoe_pupd/celebhqN2_mopoe_vae_res_beta_1__2560.0002',
                        help='mopoe model path [default: "./models/mopoe_pupd/celebhqN2_mopoe_vae_res_beta_1__2560.0002"]')
    parser.add_argument('--mvt-path', type=str, default='./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_1.0__2560.0002',
                        help='mvt model path [default: "./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_1.0__2560.0002"]')
    # parser.add_argument('--mvt-path', type=str, default='./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_0.5__2560.0002',
    #                     help='mvt model path [default: "./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_0.5__2560.0002"]')
    # parser.add_argument('--mvt-path', type=str, default='./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_0.1__2560.0002',
    #                     help='mvt model path [default: "./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_0.1__2560.0002"]')
    parser.add_argument('--mmplus-path', type=str, default='./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_1.0__2560.0002',
                        help='mmplus model path [default: "./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_1.0__2560.0002"]')
    # parser.add_argument('--mmplus-path', type=str, default='./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_0.5__2560.0002',
    #                     help='mmplus model path [default: "./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_0.5__2560.0002"]')
    # parser.add_argument('--mmplus-path', type=str, default='./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_0.1__2560.0002',
    #                     help='mmplus model path [default: "./models/mopoe_pupd/celebhqN2_mmplus_vae_res_beta_0.1__2560.0002"]')
    
    parser.add_argument('--n-comp', type=int, default=20,
                        help='ld iteration steps [default: 20]')

    parser.add_argument('--unet-vae-path', type=str, default='./models/diff/diff_vae__diffvae_hq_1000_resvaeN_openai_agn_',
                        help='unetvae path  [default: "./models/diff/diff_vae__diffvae_hq_1000_resvaeN_openai_agn_"]')

    args = parser.parse_args()

    run(args.batch_size, args.size_z1, args.size_z2, args.image_path, args.mask_path, args.attr_path, args.image_path_ae, args.mask_path_ae, args.attr_path_ae,args.unet_path, args.unet_path_ae, args.unet_openai_path, args.unet_vae_path, args.mopoe_path, args.mvt_path, args.mmplus_path)


