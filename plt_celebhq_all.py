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

from h_vae_model import ResCelebA, CelebAAttrNewBN, CelebAAttrNewBNAE
from h_vae_model_copy import ResVAEN, ResAEN
from mopoe_model import MOPOECeleb, MVTCeleb
from unet_model import Unet, UnetVAE
from unet_openai import UNetModel
from utils import *

from celeba_hq_mask_dataset import CelebAHQMaskDS
from h_vae_model import CelebAAttrNewBN
from sklearn.metrics import f1_score

new_id_to_attr = ['Bald',
        'Bangs',
        'Black_Hair',
        'Blond_Hair',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Eyeglasses',
        'Gray_Hair',
        'Heavy_Makeup',
        'Male',
        'Mouth_Slightly_Open',
        'Mustache',
        'Pale_Skin',
        'Receding_Hairline',
        'Smiling',
        'Straight_Hair',
        'Wavy_Hair',
        'Wearing_Hat',
]
    
# def get_train_test_dataloader(batch_size, size):
#     # train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
#     val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

#     # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return val_dataloader

def get_train_test_dataloader(batch_size, size):
    test_dataset = CelebAHQMaskDS(size=size, ds_type='val')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataloader

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
        losses = 0
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
            # t = t-10

        return (sample + 1) / 2


@torch.no_grad()
def evaluate_mopoe(images, masks, target, mix_vae, mix_type, amount, attr_visible, path, size_z, device, unq_name, idx, given):
    with torch.no_grad():
        mix_vae.eval()
        start_time = time.time()
        f1_avg = 0
        correct, total = 0, 0
        mask_idx = idx 
        # idx = 0

        images = images[idx-1:idx].repeat(amount,1,1,1).view(amount, *images.shape[1:]).to(device)
        masks = masks[mask_idx-1:mask_idx].repeat(amount,1,1,1).view(amount, *masks.shape[1:]).to(device)
        target = target[idx-1:idx][:,attr_visible].repeat(amount,1).view(amount,len(attr_visible)).float().to(device)
        
        present_mod_list = []

        for mod in '012':
            if mod in given:
                present_mod_list.append(int(mod))
                
        input_list = []
        if '0' in given:
            input_list.append(images)
        else:
            input_list.append(None)
        if '1' in given:
            input_list.append(masks)
        else:
            input_list.append(None)
        if '2' in given:
            input_list.append(target)
        else:
            input_list.append(None)
        
        if len(given) >= 1:
            print('mopoe present mod: ', present_mod_list, flush=True)
            all_predicted = mix_vae.cond_gen(present_mod_list, input_list)
        else:
            z = torch.normal(mean=0, std=1, size=(images.shape[0],size_z), device=device)
            all_predicted = mix_vae.sample(z)

        # print('mopoe present mod: ', present_mod_list, flush=True)
        # all_predicted = mix_vae.cond_gen(present_mod_list, input_list)
        # present_mod_list = [0,1]
        # all_predicted = mix_vae.cond_gen(present_mod_list, [images, masks, None])

        img_pred = all_predicted[0]
        mask_out = all_predicted[1]
        att_out = all_predicted[2]

        # print att out
        sigmoid_outputs = torch.sigmoid(att_out).detach().cpu()
        predicted_att = np.round(sigmoid_outputs)

        print('writing images ', flush=True)


        figure, axis = plt.subplots(4, 4)
        for row in range(4):
            for col in range(4):
                k = row*4 + col
                tar_str, pred_str = 'T: ', 'P: '
                if k > len(target) - 1:
                    break
                for ind,att in enumerate(target[k]):
                    if int(att) == 1:
                        tar_str += new_id_to_attr[ind] + '\n'
                for ind,att in enumerate(predicted_att[k]):
                    if int(att) == 1:
                        pred_str += new_id_to_attr[ind] + '\n'
                # print('predicted_att: ', pred_str)

                # plt_text = tar_str + '\n' + pred_str
                plt_text = pred_str
                px = 1/plt.rcParams['figure.dpi']
                # plt.figure(figsize=(128*px, 128*px))
                axis[row,col].text(0.1,0.1,plt_text, fontsize='xx-small', fontfamily='monospace')
                axis[row,col].axis('off')
            # break
        
        # plt.figure(figsize=(128*4*px, 128*4*px))
        plt.savefig(path['att'] + '/att_' + mix_type + '_g' + given + '_' + unq_name + '.pdf')
        print(plt_text, flush=True)

        img_grid = torchvision.utils.make_grid(img_pred, nrow=4)
        mask_grid = torchvision.utils.make_grid(mask_out, nrow=4)
        torchvision.utils.save_image(img_grid, path['image'] + '/image_' + mix_type + '_g' + given + unq_name + '.png')
        torchvision.utils.save_image(mask_grid, path['image'] + '/mask_' + mix_type + '_g' + given + unq_name + '.png')
        
        end_time = time.time()
        print(mix_type, " VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        return

@torch.no_grad()
def evaluate(images, masks, target, image_vae, mask_vae, attr_vae, attr_visible, n_mod, model, device, size_z, er, n_comp, unq_name, given, amount, path, is_vae, c, idx, unet_vae):
    losses = 0
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    model.eval()
    if is_vae:
        # sigmas = torch.tensor(np.linspace(3, 0.1, 200)).to(device) #s3e-2
        # sigmas = torch.tensor(np.linspace(10, 0.1, 500)).to(device)
        sigmas = torch.tensor(np.linspace(5, 0.1, 500)).to(device) #new
    else:
        # sigmas = torch.tensor(np.linspace(10, 0.01, 500)).to(device)
        sigmas = torch.tensor(np.linspace(5, 0.1, 500)).to(device) #new
    start_time = time.time()
    dim = int(np.sqrt(size_z))

    # idx = 1
    mask_idx = idx 
    # mask_idx = idx - 1
    images = images[idx-1:idx].repeat(amount,1,1,1).view(amount, *images.shape[1:]).to(device)
    masks = masks[mask_idx-1:mask_idx].repeat(amount,1,1,1).view(amount, *masks.shape[1:]).to(device)
    target = target[idx-1:idx][:,attr_visible].repeat(amount,1).view(amount,len(attr_visible)).float().to(device)
   
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
            z[mod] = torch.normal(mean=0, std=1, size=(amount, size_z), device=device)
        outs[mod] = models[mod].decoder(z[mod])
    
    print(given_string, flush=True)

    for s_in, s in enumerate(sigmas):
        sigma_index = torch.tensor([s_in]*amount).to(device)
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
                # outs[mod] = models[mod].decoder(z[mod])

        # for mod in given:
        #     z[mod] = z[mod] - noise[mod]

    for ind, mod in enumerate(mods):
        if mod not in given:
            outs[mod] = models[mod].decoder(z[mod])          
    
    # print att out
    sigmoid_outputs = torch.sigmoid(outs['2']).detach().cpu()
    predicted_att = np.round(sigmoid_outputs)

    print('writing images ', flush=True)

    figure, axis = plt.subplots(4, 4)
    
    for row in range(amount // 4):
        for col in range(4):
            k = row*4 + col
            if k > len(target) - 1:
                break
            tar_str, pred_str = 'T: ', 'P: '
            for ind,att in enumerate(target[k]):
                if int(att) == 1:
                    tar_str += new_id_to_attr[ind] + '\n'
            for ind,att in enumerate(predicted_att[k]):
                if int(att) == 1:
                    pred_str += new_id_to_attr[ind] + '\n'
            # print('predicted_att: ', pred_str)

            # plt_text = tar_str + '\n' + pred_str
            plt_text = pred_str
            px = 1/plt.rcParams['figure.dpi']
            # plt.figure(figsize=(128*px, 128*px))
            axis[row,col].text(0.1,0.1,plt_text, fontsize='xx-small', fontfamily='monospace')
            axis[row,col].axis('off')
        # break
    
    # plt.figure(figsize=(128*4*px, 128*4*px))
    plt.savefig(path['att'] + '/att_' + str(k) + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + '.pdf')
    print(tar_str, flush=True)
    print(plt_text, flush=True)

    diff_vae_out = evaluate_diff_vae(outs['0'], unet_vae, device)

    img_grid = torchvision.utils.make_grid(outs['0'], nrow=4)
    mask_grid = torchvision.utils.make_grid(outs['1'], nrow=4)
    diff_vae_grid = torchvision.utils.make_grid(diff_vae_out, nrow=4)

    torchvision.utils.save_image(masks[0].unsqueeze(0), path['mask'] + '/input_mask_' + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + '.png')
    torchvision.utils.save_image(images[0].unsqueeze(0), path['image'] + '/input_image_' + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + '.png')
    torchvision.utils.save_image(img_grid, path['image'] + '/img_' + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + str(c) + '.png')
    torchvision.utils.save_image(mask_grid, path['mask'] + '/mask_' + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + str(c) + '.png')
    torchvision.utils.save_image(diff_vae_grid, path['image'] + '/DIFF_img_' + '_g' + given + '_' + unq_name + str(er) + str(n_comp) + ('VAE' if is_vae else 'AE') + str(c) + '.png')

    end_time = time.time()
    print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
    return

def run(batch_size, size_z1, size_z2, image_vae_path, mask_vae_path, attr_vae_path, image_ae_path, mask_ae_path, attr_ae_path, score_vae_path, score_ae_path, score_openai_path, unet_vae_path, mopoe_model_path, mvt_model_path, unq_name):
    res_size = 128
    unq_name = unq_name + str(int(torch.rand(1)*10000))
    print('vars: ', batch_size, size_z1, size_z2, unq_name, flush=True)
    train_losses, val_losses = [], []
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    print(attr_visible, flush=True)

    path = {'image': './samples/celeb_scoreAE/', 'mask': './samples/celeb_scoreAE/', 'att': './samples/celeb_scoreAE/'}
    for p in path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:2")

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

    # MoPoE model
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

    n_mod = 3
    # score_model = Unet(dim=16, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True)
    score_model = Unet(dim=128, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True) # New
    score_model.load_state_dict(torch.load(score_vae_path, map_location=device)['model_state_dict'])
    score_model = score_model.to(device)

    # score_model_ae = Unet(dim=16, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True)
    score_model_ae = Unet(dim=128, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True) # New
    # score_model_ae = CAUNET(n_mod=n_mod, z_dim=size_z1, dim=16, dim2=16, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True, cross=True)
    score_model_ae.load_state_dict(torch.load(score_ae_path, map_location=device)['model_state_dict'])
    score_model_ae = score_model_ae.to(device)

    score_openai = UNetModel(in_channels=n_mod, model_channels=128, out_channels=n_mod, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, channel_mult=(1,2,2,2,2), num_heads=8)
    score_openai.load_state_dict(torch.load(score_openai_path, map_location=device)['model_state_dict'])
    score_openai = score_openai.to(device)
    
    unetvae = UNetModel(in_channels=img_ch*2, model_channels=128, out_channels=img_ch, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, channel_mult=(1,2,2,3,4), num_heads=8)
    # unetvae = UnetVAE(dim=128, channels=3, dim_mults=(1,2,2,2,4), with_time_emb=True)
    unetvae.load_state_dict(torch.load(unet_vae_path, map_location=device)['model_state_dict'])
    unetvae = unetvae.to(device)

    print('Models loaded', flush=True)
    
    val_dataloader = get_train_test_dataloader(batch_size, res_size)
    print('data loaded ', flush=True)

    images, masks, target = next(iter(val_dataloader))

    # img_grid = torchvision.utils.make_grid(images, nrow=16)
    # torchvision.utils.save_image(img_grid, path['image'] + 'input_image.png')


    # a sample from validation set
    idx = 120 - 3

    unq_name_ae = unq_name + 'ae'
    unq_name_mopoe = unq_name + 'mopoe'

    # given = '0'
    given = '2'
    amount = 16

    n_comp = 2
    n_comp_ae = 2

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
        c_vae = {'0': 0.8, '1': 0.5, '2': 0}
        c_ae = {'0': 0.8, '1': 0.5, '2': 0.5}
        er = {'0': 0.018, '1': 0.015, '2': 0.015}
        er_ae = {'0': 0.02, '1': 0.015, '2': 0.015}

    

    print('ncomp: ', n_comp, n_comp_ae, ' er: ', er, er_ae, 'c: ', c_vae, c_ae, ' amount: ', amount)

    print('AE') # AE
    evaluate(images, masks, target, image_ae, mask_ae, attr_ae, attr_visible, n_mod, score_model_ae, device, size_z1, er_ae, n_comp_ae, unq_name_ae, given, amount, path, False, c_ae, idx, unetvae) 
    # print('OPENAI AE') # openaiAE
    # evaluate(images, masks, target, image_ae, mask_ae, attr_ae, attr_visible, n_mod, score_openai, device, size_z1, er_ae, n_comp_ae, unq_name_ae + 'OPENAI', given, amount, path, False, c_ae, idx, unetvae) 
    print('VAE') # VAE
    evaluate(images, masks, target, image_vae, mask_vae, attr_vae, attr_visible, n_mod, score_model, device, size_z1, er, n_comp, unq_name, given, amount, path, True, c_vae, idx, unetvae)
    print('MoPoE') # MoPoE
    evaluate_mopoe(images, masks, target, mopoe_vae, 'mopoe', amount, attr_visible, path, size_z1, device, unq_name_mopoe, idx, given)
    print('MVTCAE') # MVTCAE
    evaluate_mopoe(images, masks, target, mvt_vae, 'mvt', amount, attr_visible, path, size_z1, device, unq_name_mopoe, idx, given)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z1', type=int, default=256,
                        help='size of z1 [default: 256]')
    parser.add_argument('--size-z2', type=int, default=256,
                        help='size of z2 [default: 256]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--unq-name', type=str, default='plt_all22c05',
                        help='name to identify the model [default: "plt_allc05"]')
    
    parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__',
                        help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__"]')
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.5_smN_256__',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.5_smN_256__"]')
    parser.add_argument('--mask-path', type=str, default='./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq',
                        help='mask vae model path [default: "./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq"]')
    parser.add_argument('--attr-path', type=str, default='./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1',
                        help='vae model path [default: "./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1"]')
    
    # parser.add_argument('--unet-path', type=str, default='./models/celeb_score/celeb_hq_unetVAE_256_uhqVAE_b01_s3e2_18_uhqVAE_b01_s3e2_',
    #                     help='score path  [default: "./models/celeb_score/celeb_hq_unetVAE_256_uhqVAE_b01_s3e2_18_uhqVAE_b01_s3e2_"]')
    # parser.add_argument('--unet-path', type=str, default='./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEb05_s10_01_18_uhqVAEb05_s10_01_',
    #                     help='score path  [default: "./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEb05_s10_01_18_uhqVAEb05_s10_01_"]')
    # parser.add_argument('--unet-path', type=str, default='./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEC3b01_z256_dim128_12222_s5_01_18_uhqVAEC3b01_z256_dim128_12222_s5_01_',
    #                     help='score path  [default: "./models/celeb_score/celeb_hq_unetVAE_256_uhqVAEC3b01_z256_dim128_12222_s5_01_18_uhqVAEC3b01_z256_dim128_12222_s5_01_"]')
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

    parser.add_argument('--unet-vae-path', type=str, default='./models/diff/diff_vae__diffvae_hq_1000_resvaeN_openai_agn_',
                        help='unetvae path  [default: "./models/diff/diff_vae__diffvae_hq_1000_resvaeN_openai_agn_"]')
    # parser.add_argument('--unet-vae-path', type=str, default='./models/diff/diff_vae__diffvae_hq_1000_b1_',
    #                     help='unetvae path  [default: "./models/diff/diff_vae__diffvae_hq_1000_b1_"]')
    
    parser.add_argument('--unet-openai-path', type=str, default='./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_18_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_',
                        help='openai score path  [default: "./models/celeb_score/celeb_hq_unet_AE_256_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_18_uhqAEnew_z256_dim128_OPENAI_12222_s5_01_"]')
   
    parser.add_argument('--mopoe-path', type=str, default='./models/mopoe_pupd/celebhqN2_mopoe_vae_res_beta_1__2560.0002',
                        help='mopoe model path [default: "./models/mopoe_pupd/celebhqN2_mopoe_vae_res_beta_1__2560.0002"]')
    parser.add_argument('--mvt-path', type=str, default='./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_1.0__2560.0002',
                        help='mvt model path [default: "./models/mopoe_pupd/celebhqN2_mvt_vae_res_beta_1.0__2560.0002"]')
    
    parser.add_argument('--n-comp', type=int, default=20,
                        help='ld iteration steps [default: 20]')

    args = parser.parse_args()

    run(args.batch_size, args.size_z1, args.size_z2, args.image_path, args.mask_path, args.attr_path, args.image_path_ae, args.mask_path_ae, args.attr_path_ae,args.unet_path, args.unet_path_ae, args.unet_openai_path, args.unet_vae_path, args.mopoe_path, args.mvt_path, args.unq_name)