import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"

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

from h_vae_model_copy import ResVAEN, ResAEN
from unet_model import CAUNET, Unet, UnetNodown
from unet_openai import UNetModel
from lat_sm2_model import ClwithTime2, ClwithTime3
from utils import *

from celeba_hq_mask_dataset import CelebAHQMaskDS
from h_vae_model import CelebAAttrNewBN, CelebAAttrNewBNAE
from sklearn.metrics import f1_score

from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil
from sde_helper2 import *

from configs import new_id_to_attr
    
def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader

def get_val_dataloader(batch_size, size):
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_dataloader

def get_test_dataloader(batch_size, size):
    test_dataset = CelebAHQMaskDS(size=size, ds_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataloader


def train_model(train_loader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, sm_model, size_z, optimizer, device, sde, likelihood_weighting=True, vae_type="VAE", im_sample=False, reparametrize=False):
    losses = 0
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    sm_model.train()

    start_time = time.time()
    dim = int(np.sqrt(size_z))

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = images.to(device)
        masks = masks.to(device)
        target = target.to(device)[:, attr_visible]

        with torch.no_grad():
            # Get z for each modality
            if vae_type == "VAE":
                image_mu, image_logvar = image_vae.encoder(input)
                mask_mu, mask_logvar = mask_vae.encoder(masks)
                attr_mu, attr_logvar = attr_vae.encoder(target.float())

                if reparametrize:
                    z_image = image_vae.reparametrize(image_mu, image_logvar)
                    z_mask = mask_vae.reparametrize(mask_mu, mask_logvar)
                    z_attr = attr_vae.reparametrize(attr_mu, attr_logvar)
                else:
                    z_image = image_mu
                    z_mask =  mask_mu
                    z_attr = attr_mu
            elif vae_type == "AE":
                z_image = image_vae.encoder(input)
                z_mask = mask_vae.encoder(masks)
                z_attr = attr_vae.encoder(target.float())
            

        with torch.enable_grad():
            z = torch.cat([z_image.unsqueeze(1), z_mask.unsqueeze(1), z_attr.unsqueeze(1)], dim=1).view(-1,n_mod,dim,dim)
            loss = loss_fn(z, sm_model, sde, reduce_mean=True, likelihood_weighting=likelihood_weighting, eps=1e-5, im_sample=im_sample)
            losses += loss.item()      

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    losses /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return losses

@torch.no_grad()
def evaluate(val_loader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, sm_model, size_z, device, sde, epoch, unq_name, save_paths, likelihood_weighting=True, eps=1e-3, noise_obs=False, vae_type="VAE", pc=False, n_steps=1, target_snr=0.16, im_sample=False, cl_g=None, cl_s=None, reparametrize=False):
    losses = 0
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    sm_model.eval()
    
    start_time = time.time()
    dim = int(np.sqrt(size_z))

    for batch_idx, (images, masks, target) in enumerate(val_loader):

        input = images.to(device)
        masks = masks.to(device)
        target = target.to(device)[:, attr_visible]
        

        if vae_type == "VAE":
            image_mu, image_logvar = image_vae.encoder(input)
            mask_mu, mask_logvar = mask_vae.encoder(masks)
            attr_mu, attr_logvar = attr_vae.encoder(target.float())

            if reparametrize:
                z_image = image_vae.reparametrize(image_mu, image_logvar)
                z_mask = mask_vae.reparametrize(mask_mu, mask_logvar)
                z_attr = attr_vae.reparametrize(attr_mu, attr_logvar)
            else:
                z_image = image_mu
                z_mask =  mask_mu
                z_attr = attr_mu
        elif vae_type == "AE":
            z_image = image_vae.encoder(input)
            z_mask = mask_vae.encoder(masks)
            z_attr = attr_vae.encoder(target.float())

        z = torch.cat([z_image.unsqueeze(1), z_mask.unsqueeze(1), z_attr.unsqueeze(1)], dim=1).view(-1,n_mod,dim,dim)
        loss = loss_fn(z, sm_model, sde, reduce_mean=True, likelihood_weighting=likelihood_weighting, eps=1e-5, im_sample=im_sample)
        losses += loss.item()      
        

    if (epoch > 20) and ((epoch+1) % 10 == 0):
        mods = '012' # 0 for image, 1 for mask, 2 for attr
        z = {}
        amount = 1
        models = {'0': image_vae, '1': mask_vae, '2': attr_vae}
        samples = {'0': input[0].unsqueeze(0), '1': masks[0].unsqueeze(0), '2': target[0].unsqueeze(0).float()}
        target_clg = torch.ones(samples['0'].shape[0], 1).to(device)
        outs = {}
        noised = {}
        given = '12'

        for mod in mods:
            if mod in given:
                if vae_type == "VAE":
                    if reparametrize:
                        z[mod] = models[mod].reparametrize(*models[mod].encoder(samples[mod]))
                    else:
                        z[mod] = models[mod].encoder(samples[mod])[0]
                elif vae_type == "AE":
                    z[mod] = models[mod].encoder(samples[mod])
            else:
                z[mod] = sde.prior_sampling((1,size_z)).to(device)
            outs[mod] = models[mod].decoder(z[mod])
        
        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(z[mods[0]].shape[0], device=t.device) * t

            for mod in mods:
                if noise_obs:
                    if mod in given:
                        mean, std = sde.marginal_prob(z[mod].view(-1,1,dim,dim), vec_t)
                        noised[mod] = (mean + std[:, None, None, None] * z[mod].view(-1,1,dim,dim)).view(-1, size_z)
                    else:
                        noised[mod] = z[mod]
                else:
                    noised[mod] = z[mod]

            z_upd = torch.cat([noised[mod].unsqueeze(1) for mod in mods], dim=1).view(-1,n_mod,dim,dim).detach()
            if pc:
                z_upd, z_mean = corrector(z_upd, vec_t, sm_model, sde, n_steps, target_snr, cl_g=cl_g, cl_s=cl_s, target=target_clg, given=given, all_mods=mods)
            z_upd, z_mean = em_predictor(z_upd, vec_t, sm_model, sde, cl_g=cl_g, cl_s=cl_s, target=target_clg, given=given, all_mods=mods)
             
            for ind,mod in enumerate(mods):
                if mod not in given:
                    z[mod] =  z_upd[:,ind].view(amount,size_z)

        for ind,mod in enumerate(mods):
            if mod not in given:
                z[mod] =  z_mean[:,ind].view(amount,size_z)

        for mod in mods:
            outs[mod] = models[mod].decoder(z[mod])

        sigmoid_outputs = torch.sigmoid(outs['2']).detach().cpu()
        predicted_att = np.round(sigmoid_outputs)

        tar_str, pred_str = 'T: ', 'P: '
        for ind,att in enumerate(target[0]):
            if int(att) == 1:
                tar_str += new_id_to_attr[ind] + ' '
        for ind,att in enumerate(predicted_att[0]):
            if int(att) == 1:
                pred_str += new_id_to_attr[ind] + ' '

        plt.figure()
        grid = torchvision.utils.make_grid(torch.cat([samples['0'], outs['0']],dim=0), nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig(save_paths['images'] + 'img_' + str(epoch) + '_' +  unq_name + '.png')
        plt.figure()
        grid = torchvision.utils.make_grid(torch.cat([samples['1'], outs['1']],dim=0), nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig(save_paths['images'] + '_mask_' + str(epoch) + '_' +  unq_name + '.png')
        plt.figure()
        plt.text(0.05,0.5,tar_str + '\n' + pred_str)
        plt.savefig(save_paths['images'] + '_att_' + str(epoch) + '_' +  unq_name + '.png')   
        plt.close('all') 

    end_time = time.time()
    losses /= len(val_loader)
    print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
    print("Validation loss: ", losses, flush=True)
    return losses

@torch.no_grad()
def calc_perf(val_loader, image_vae, mask_vae, attr_vae, attr_visible, att_threshold, n_mod, sm_model, device, sde, size_z, given, path, eps=1e-3, noise_obs=False, vae_type="VAE", pc=False, n_steps=1, target_snr=0.16, cl_g=None, cl_s=None, reparametrize=False):
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    sm_model.eval()
    
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

        target_clg = torch.ones(images.shape[0], 1).to(device)

        mods = '012' # 0 for image, 1 for mask, 2 for attr
        z = {}
        models = {'0': image_vae, '1': mask_vae, '2': attr_vae}
        samples = {'0': images, '1': masks, '2': target.float()}
        mod_to_word = {'0': 'IMAGE', '1': 'MASK', '2': 'TARGET'}
        outs = {}
        noised = {}

        given_string = "GIVEN "

        for mod in mods:
            if mod in given:
                if vae_type == "VAE":
                    if reparametrize:
                        z[mod] = models[mod].reparametrize(*models[mod].encoder(samples[mod]))
                    else:
                        z[mod] = models[mod].encoder(samples[mod])[0]
                elif vae_type == "AE":
                    z[mod] = models[mod].encoder(samples[mod])
                given_string += mod_to_word[mod] + " "
            else:
                z[mod] = sde.prior_sampling((images.shape[0], size_z)).to(device)
            outs[mod] = models[mod].decoder(z[mod])

        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(z[mods[0]].shape[0], device=t.device) * t

            for mod in mods:
                if noise_obs:
                    if mod in given:
                        mean, std = sde.marginal_prob(z[mod].view(-1,1,dim,dim), vec_t)
                        noised[mod] = (mean + std[:, None, None, None] * z[mod].view(-1,1,dim,dim)).view(-1, size_z)
                    else:
                        noised[mod] = z[mod]
                else:
                    noised[mod] = z[mod]

            z_upd = torch.cat([noised[mod].unsqueeze(1) for mod in mods], dim=1).view(-1,n_mod,dim,dim).detach()
            
            z_upd, z_mean = em_predictor(z_upd, vec_t, sm_model, sde, cl_g=cl_g, cl_s=cl_s, target=target_clg, given=given, all_mods=mods)
            if pc:
                z_upd, z_mean = corrector(z_upd, vec_t, sm_model, sde, n_steps, target_snr, cl_g=cl_g, cl_s=cl_s, target=target_clg, given=given, all_mods=mods)
                        
            for ind,mod in enumerate(mods):
                if mod not in given:
                    z[mod] =  z_upd[:,ind].view(images.shape[0],size_z)

        
        for ind,mod in enumerate(mods):
            if mod not in given:
                z[mod] =  z_mean[:,ind].view(images.shape[0],size_z)

        for mod in mods:
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
        
        if '0' not in given:
            # Calc Fid Image
            save_batch_image(images, path['in_image'] + str(batch_idx) + '_')
            if vae_type == "VAE":
                save_batch_image(outs['0'], path['out_image_vae'] + str(batch_idx) + '_')
            else:
                save_batch_image(outs['0'], path['out_image_ae'] + str(batch_idx) + '_')

        # print('done one batch!', flush=True)

    print(given_string, flush=True)
    if '0' not in given:
        print('calculating FID', flush=True)
        if vae_type == "VAE":
            fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_vae']], 256, device, 2048, 2)
        else:
            fid_img = calculate_fid_given_paths([path['in_image'], path['out_image_ae']], 256, device, 2048, 2)

        print("Image FID: ", fid_img, flush=True)
    
    f1_mask = f1_score(torch.cat(true_mask, dim=0).numpy(), torch.cat(predicted_mask, dim=0).numpy(), average='samples')
    f1_att = f1_score(torch.cat(true_att, dim=0).numpy(), torch.cat(predicted_att, dim=0).numpy(), average='samples')
    end_time = time.time()

    print("SM-UNET VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
    
    print("SM-UNET F1 Attribute score: ", f1_att, flush=True)
    print("SM-UNET Acc Attribute: ", correct_att/total_att)

    print("SM-UNET F1 Mask score: ", f1_mask, flush=True)
    print("SM-UNET Acc Mask: ", correct_mask/total_mask)

    return


def run(epochs, batch_size, lr, size_z1, size_z2, image_model_path, mask_model_path, attr_model_path, unq_name, cuda_num, vae_type, sde_type, beta_0, beta_1, N, T, likelihood_weighting, noise_obs, pc, n_steps, target_snr, im_sample, use_clg, clg_path, cl_s, eval_only, score_path, reparametrize, test_set):
    res_size = 128
    print('\n vars: ', epochs, batch_size, lr, size_z1, size_z2, unq_name, flush=True)
    train_losses, val_losses = [], []
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    print(attr_visible, flush=True)

    path = {'model': './models/celeb_cont/', 'plots': './plots/celeb_cont/', 'images': './images/celeb_cont/'}
    for p in path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    rand_num = str(int(torch.rand(1)*10000))
    temp_dir_name = './t_' + str(unq_name) + '_' + str(rand_num) + '/'
    print('temp dir: ', temp_dir_name, flush=True)

    sample_path = {'in_image': temp_dir_name + 'temp_hq_in' + rand_num + '/', 
            'out_image_vae': temp_dir_name + 'temp_hq_out_vae' + rand_num + '/', 
            'out_image_ae': temp_dir_name + 'temp_hq_out_ae' + rand_num + '/', }
    
    for p in sample_path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, cuda_num, flush=True)
    device = torch.device("cuda:" + str(cuda_num) if cuda else "cpu")

    likelihood_weighting = True if likelihood_weighting else False
    noise_obs = True if noise_obs else False
    im_sample = True if im_sample else False
    pc = True if pc else False
    use_clg = True if use_clg else False
    eval_only = True if eval_only else False
    reparametrize = True if reparametrize else False
    test_set = True if test_set else False
    att_threshold = 0.5

    print("SDE: ", sde_type, " likelihood_weighting: ", likelihood_weighting, " imp: ", im_sample, " T: ", T, " beta0: ", beta_0, " beta1: ", beta_1, " N: ", N, " noise_obs: ", noise_obs, " VAE type: ", vae_type, " pc: ", pc, " snr: ", target_snr, " n-steps: ", n_steps, flush=True)
    print("use clg: ", use_clg, " cl_scale: ", cl_s, flush=True)
    if eval_only:
        print("Eval only: ", eval_only, " score path: ", score_path, flush=True)
        if use_clg:
            print("Clg path: ", clg_path, flush=True)
        else:
            print("No Clg", flush=True)

    if test_set:
        print("Test SET", flush=True)

    # Load mask model
    enc_channel_list = [(64,128,128,4), (128,256,256,4)]
    dec_channel_list = [(256,256,128,4), (128,128,64,4)]
    size_in = res_size
    mask_img_ch = 1
    if vae_type == "VAE":    
        mask_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z1, mask_img_ch)
    elif vae_type == "AE":
        mask_vae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z1, mask_img_ch)
    else:
        raise Exception("Wrong VAE type")
    mask_vae.load_state_dict(torch.load(mask_model_path, map_location=device)['model_state_dict'])
    mask_vae = mask_vae.to(device)

    # Load image model
    #sm
    enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
    dec_channel_list = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
    # enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2), (512,1024,1024,2)]
    # dec_channel_list = [(1024,1024,512,2), (512,512,256,2), (256,256,128,2), (128,128,64,2)]
    size_in = res_size
    img_ch = 3  
    if vae_type == "VAE":  
        image_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    elif vae_type == "AE":
        image_vae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z1, img_ch)
    else:
        raise Exception("Wrong VAE type")
    image_vae.load_state_dict(torch.load(image_model_path, map_location=device)['model_state_dict'])
    image_vae = image_vae.to(device)

    # Load attr model
    if vae_type == "VAE":
        attr_vae = CelebAAttrNewBN(size_z2)
    elif vae_type == "AE":
        attr_vae = CelebAAttrNewBNAE(size_z2)
    else:
        raise Exception("Wrong VAE type")
    attr_vae.load_state_dict(torch.load(attr_model_path, map_location=device)['model_state_dict'])
    attr_vae = attr_vae.to(device)

    assert size_z1 == size_z2

    n_mod = 3
    # dim = 128
    dim = 256
    score_model = Unet(dim=dim, channels=n_mod, dim_mults=(1,2,2,2,2), with_time_emb=True)
    # score_model = Unet(dim=dim, channels=n_mod, dim_mults=(1,2,4,8), with_time_emb=True)
    # score_model = UNetModel(in_channels=n_mod, model_channels=128, out_channels=n_mod, num_res_blocks=2, attention_resolutions=(8,16,), dropout=0.1, channel_mult=(1,2,4,8), num_heads=8)

    if not eval_only:
        optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
        score_model = score_model.to(device)
    else:
        score_model.load_state_dict(torch.load(score_path, map_location=device)['model_state_dict'])
        score_model = score_model.to(device)
        score_model.eval()

    if use_clg:
        cl_model = {}
        for mods in ['01', '02', '12']:
            cl_model[mods] = ClwithTime2(n_mod=2, size_z=size_z1, n_class=1)
            cl_model[mods].load_state_dict(torch.load(clg_path[mods], map_location=device)['model_state_dict'])
            cl_model[mods] = cl_model[mods].to(device)
            cl_model[mods].eval()
            print("classfier model ", mods, " loaded ", flush=True)
    else:
        cl_model = None

    if sde_type == "VPSDE":
        print("Initializing VPSDE", flush=True)
        sde = VPSDE(beta_min=beta_0, beta_max=beta_1, N=N)
    elif sde_type == "VESDE":
        print("Initializing VESDE", flush=True)
        sde = VESDE(sigma_min=beta_0, sigma_max=beta_1, N=N)
    elif sde_type == "subVPSDE":
        print("Initializing subVPSDE", flush=True)
        sde = subVPSDE(beta_min=beta_0, beta_max=beta_1, N=N)

    unq_name += sde_type + str(size_z1) + '_vtype_' + vae_type + '_dim_' + str(dim) + '_N_' + str(sde.N) + '_b_' + str(sde.beta_0) + '_' + str(sde.beta_1) + '_'
    if likelihood_weighting:
        unq_name += '_ll_'
    if likelihood_weighting and im_sample:
        unq_name += '_ImpSamp_'
    if noise_obs:
        unq_name += '_n_obs_'
    if pc:
        unq_name += '_pc_' + str(pc) + '_snr_' + str(target_snr) + '_'

    print("unq_name: ", unq_name, flush=True)
    
    if not eval_only:

        train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, res_size)
        print("data loaded ", flush=True)

        for epoch in range(epochs):
            print("Epoch: "+str(epoch + 1), flush=True)

            training_loss = train_model(train_dataloader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, score_model, size_z1, optimizer, device, sde, likelihood_weighting=likelihood_weighting, vae_type=vae_type, im_sample=im_sample, reparametrize=reparametrize)
            validation_loss = evaluate(val_dataloader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, score_model, size_z1, device, sde, epoch, unq_name, path, likelihood_weighting=likelihood_weighting, eps=1e-3, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, im_sample=im_sample, cl_g=cl_model, cl_s=cl_s, reparametrize=reparametrize)
            print(' ', flush=True)

            train_losses.append(training_loss)
            val_losses.append(validation_loss)

            if epoch == 0:
                prev_loss = validation_loss
            if epoch > 0 and (validation_loss < prev_loss):
                torch.save({
                'epoch': epoch,
                'model_state_dict': score_model.state_dict(),
                'train_loss': training_loss,
                'val_loss': validation_loss,
                'size_z': size_z1,
                }, path['model'] + "celeb_hq_cont_" + str(size_z1) + str(unq_name) + str(len(attr_visible)))
                print('Model saved', flush=True)
                prev_loss = validation_loss

            # if (epoch + 1) % 50 == 0:
            #     lr /= 5
            #     optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

            # if (epoch + 1) == epochs:
            torch.save({
            'epoch': epoch,
            'model_state_dict': score_model.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'size_z': size_z1,
            }, path['model'] + "celeb_hq_cont_" + str(size_z1) + str(unq_name) + str(len(attr_visible)) + '_last_epoch')
            print('Last Model saved', flush=True)

            if (epoch + 1) % 500 == 0:
                for given in ['', '0', '1', '2', '01', '02', '12']:
                    calc_perf(val_dataloader, image_vae, mask_vae, attr_vae, attr_visible, att_threshold, n_mod, score_model, device, sde, size_z1, given, sample_path, eps=1e-3, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s, reparametrize=reparametrize)


        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], path['plots'] + 'celeb_hq_cont_' + '_' + unq_name)

    else:
        if not test_set:
            val_dataloader = get_val_dataloader(batch_size, res_size)
        else:
            val_dataloader = get_test_dataloader(batch_size, res_size)
        print("data loaded ", flush=True)

        validation_loss = evaluate(val_dataloader, image_vae, mask_vae, attr_vae, attr_visible, n_mod, score_model, size_z1, device, sde, 1, unq_name, path, likelihood_weighting=likelihood_weighting, eps=1e-3, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, im_sample=im_sample, cl_g=cl_model, cl_s=cl_s, reparametrize=reparametrize)
        print(' ', flush=True)

        for given in ['', '0', '1', '2', '01', '02', '12']:
        # for given in ['12']:
            # for cl_s in [0, 1, 10, 100, 250, 500, 1000]:
            # for cl_s in [0, 1, 10, 100, 1000, 10000, 30000, 50000]:
            # for cl_s in [1000, 1200, 1500, 1800, 2000, 2500, 3000]:
            #     print("cl_s: ", cl_s, flush=True)
            # for target_snr in [1e-5, 1e-3, 0, 1, 10, 100]:
            #        print("target snr: ", target_snr)
            calc_perf(val_dataloader, image_vae, mask_vae, attr_vae, attr_visible, att_threshold, n_mod, score_model, device, sde, size_z1, given, sample_path, eps=1e-3, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s, reparametrize=reparametrize)


    shutil.rmtree(temp_dir_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z1', type=int, default=256,
                        help='size of z1 [default: 256]')
    parser.add_argument('--size-z2', type=int, default=256,
                        help='size of z2 [default: 256]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train [default: 1000]')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate [default: 0.00005]')
    parser.add_argument('--unq-name', type=str, default='cel_sde',
                        help='name to identify the model [default: "cel_sde"]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_1.0_smN_256__',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_1.0_smN_256__"]')
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.5_smN_256__',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.5_smN_256__"]')
    
    ## 256 VAE
    parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__',
                        help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__"]')
    parser.add_argument('--mask-path', type=str, default='./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq',
                        help='mask vae model path [default: "./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq"]')
    parser.add_argument('--attr-path', type=str, default='./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1',
                        help='vae model path [default: "./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1"]')
    
    # # 256 AE
    parser.add_argument('--image-path-ae', type=str, default='./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_',
                        help='image path for ae [default: "./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_"]')
    parser.add_argument('--mask-path-ae', type=str, default='./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_',
                        help='mask path for ae [default: "./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_"]')
    parser.add_argument('--attr-path-ae', type=str, default='./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1',
                        help='attr path for ae [default: "./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1"]')

    # #1024 VAE
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_1024_beta_0.1_smN_',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_1024_beta_0.1_smN_"]')
    # parser.add_argument('--mask-path', type=str, default='./models/celeba_mask/celeb_hq_mask_dsize_128_z_1024_beta_1_mask_celebhq',
    #                     help='mask vae model path [default: "./models/celeba_mask/celeb_hq_mask_dsize_128_z_1024_beta_1_mask_celebhq"]')
    # parser.add_argument('--attr-path', type=str, default='./models/celeba_attr/celeba_attr_bn_hq__z_1024_beta_0.1',
    #                     help='vae model path [default: "./models/celeba_attr/celeba_attr_bn_hq__z_1024_beta_0.1"]')

    # # 1024 AE
    # parser.add_argument('--image-path-ae', type=str, default='./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_10241024_re4ne3_sm_hq_',
    #                     help='image path for ae [default: "./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_10241024_re4ne3_sm_hq_"]')
    # parser.add_argument('--mask-path-ae', type=str, default='./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_1024mask_hq1024_re5ne3__',
    #                     help='mask path for ae [default: "./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_1024mask_hq1024_re5ne3__"]')
    # parser.add_argument('--attr-path-ae', type=str, default='./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_1024_0.0001att_hq1024_re4ne1__',
    #                     help='attr path for ae [default: "./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_1024_0.0001att_hq1024_re4ne1__"]')


    parser.add_argument('--vae-type', type=str, default='VAE',
                        help='vae type: AE or VAE [default: "VAE"]')
    parser.add_argument('--sde-type', type=str, default='VPSDE',
                        help='sde type: VPSDE, VESDE, or subVPSDE [default: "VPSDE"]')
    parser.add_argument('--reparametrize', type=int, default=0, 
                        help='If 1, sample from vae else use mean')
    parser.add_argument('--beta0', type=float, default=0.1,
                        help='beta0  [default: 0.1]')
    parser.add_argument('--beta1', type=float, default=20,
                        help='beta1  [default: 20]')
    parser.add_argument('--N', type=int, default=100,
                        help='Number of iterations [default: 100]')
    parser.add_argument('--T', type=int, default=1,
                        help='Max Timestep [default: 1]')
    parser.add_argument('--ll-weighting', type=int, default=0, 
                        help='if 1, likelihood weighting=True else False')
    parser.add_argument('--noise-obs', type=int, default=1, 
                        help='if 1, add noise to observed variables')
    parser.add_argument('--im-sample', type=int, default=0, 
                        help='if 1, use importance sampling for likelihood weighting')
    parser.add_argument('--pc', type=int, default=0, 
                        help='if 1, use langevin corrector')
    parser.add_argument('--n-steps', type=int, default=1, 
                        help='langevin step')
    parser.add_argument('--target-snr', type=float, default=0.16,
                        help='target signal to noise ratio used in langevin step  [default: 0.16]')
    
    parser.add_argument('--use-clg', type=int, default=0, 
                        help='if 1, use classifier guidance')
    # parser.add_argument('--clg-path', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time3__vtype_VAE_b_0.1_20.0_',
    #                     help='classifier guidance path [default: "./models/cel_clf_time/256cel_sde_cls_with_time3__vtype_VAE_b_0.1_20.0_"]')
    # parser.add_argument('--clg-path', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time3__vtype_VAEMODS_02__b_0.1_20.0_',
    #                     help='classifier guidance path [default: "./models/cel_clf_time/256cel_sde_cls_with_time3__vtype_VAEMODS_02__b_0.1_20.0_"]')
    parser.add_argument('--clg-path-01', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_01__b_0.1_20.0_',
                        help='classifier guidance path of 01 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_01__b_0.1_20.0_"]')
    parser.add_argument('--clg-path-02', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_02__b_0.1_20.0_',
                        help='classifier guidance path of 02 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_02__b_0.1_20.0_"]')
    parser.add_argument('--clg-path-12', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_12__b_0.1_20.0_',
                        help='classifier guidance path of 12 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_VAEMODS_12__b_0.1_20.0_"]')

    parser.add_argument('--clg-path-ae-01', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_01__b_0.1_20.0_',
                        help='classifier guidance path of 01 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_01__b_0.1_20.0_"]')
    parser.add_argument('--clg-path-ae-02', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_02__b_0.1_20.0_',
                        help='classifier guidance path of 02 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_02__b_0.1_20.0_"]')
    parser.add_argument('--clg-path-ae-12', type=str, default='./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_12__b_0.1_20.0_',
                        help='classifier guidance path of 12 [default: "./models/cel_clf_time/256cel_sde_cls_with_time_EBM_NOIND__vtype_AEMODS_12__b_0.1_20.0_"]')
    

    parser.add_argument('--cl-s', type=float, default=1.0,
                        help='classifier guidance scale  [default: 1.0]')
    
    parser.add_argument('--eval-only', type=int, default=0, 
                        help='if 1, no training, eval only')
    parser.add_argument('--test-set', type=int, default=0, 
                        help='if 1, use testset, else val test')
    # parser.add_argument('--score-path', type=str, default='./models/celeb_cont/celeb_hq_cont_256cel_sde_vtype_VAE_dim_128_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18',
    #                     help='score path [default: "./models/celeb_cont/celeb_hq_cont_256cel_sde_vtype_VAE_dim_128_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18"]')
    parser.add_argument('--score-path', type=str, default='./models/celeb_cont/celeb_hq_cont_256cel_sde_VPSDE256_vtype_VAE_dim_256_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18_last_epoch',
                        help='score path [default: "./models/celeb_cont/celeb_hq_cont_256cel_sde_VPSDE256_vtype_VAE_dim_256_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18_last_epoch"]')
    parser.add_argument('--score-path-ae', type=str, default='./models/celeb_cont/celeb_hq_cont_256cel_sdeVPSDE256_vtype_AE_dim_256_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18_last_epoch',
                        help='score path-ae [default: "./models/celeb_cont/celeb_hq_cont_256cel_sdeVPSDE256_vtype_AE_dim_256_N_1000_b_0.1_20.0__n_obs__pc_True_snr_0.16_18_last_epoch"]')


    args = parser.parse_args()

    if args.vae_type == "VAE":
        clg_paths = {}
        clg_paths['01'] = args.clg_path_01
        clg_paths['02'] = args.clg_path_02
        clg_paths['12'] = args.clg_path_12
        run(args.epochs, args.batch_size, args.lr, args.size_z1, args.size_z2, args.image_path, args.mask_path, args.attr_path, args.unq_name, args.cuda, \
            args.vae_type, args.sde_type, args.beta0, args.beta1, args.N, args.T, args.ll_weighting, args.noise_obs, args.pc, args.n_steps, args.target_snr, args.im_sample, args.use_clg, clg_paths, args.cl_s, args.eval_only, args.score_path, args.reparametrize, args.test_set)
    elif args.vae_type == "AE":
        clg_paths = {}
        clg_paths['01'] = args.clg_path_ae_01
        clg_paths['02'] = args.clg_path_ae_02
        clg_paths['12'] = args.clg_path_ae_12
        run(args.epochs, args.batch_size, args.lr, args.size_z1, args.size_z2, args.image_path_ae, args.mask_path_ae, args.attr_path_ae, args.unq_name, args.cuda, \
            args.vae_type, args.sde_type, args.beta0, args.beta1, args.N, args.T, args.ll_weighting, args.noise_obs, args.pc, args.n_steps, args.target_snr, args.im_sample, args.use_clg, clg_paths, args.cl_s, args.eval_only, args.score_path_ae, args.reparametrize, args.test_set)


