import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp

from h_vae_model_copy import ResVAEN, ResAEN
from celeba_hq_mask_dataset import CelebAHQMaskDS
from h_vae_model import CelebAAttrNewBN, CelebAAttrNewBNAE
from lat_sm2_model import ClwithTime3

from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil

from utils import *

def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader

# Taken and updated from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py

def marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    # print('marginal prob x.shape: ',x.shape, 't: ', t.shape, 'logmean: ', log_mean_coeff[:, None, None, None].shape, flush=True)
    mean = torch.exp(log_mean_coeff.view(-1, *([1]*len(x.shape[1:])))) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def likelihood_importance_cum_weight(t, beta_0, beta_1, eps=1e-5):
    exponent1 = 0.5 * eps * (eps - 2) * beta_0 - 0.5 * eps ** 2 * beta_1
    exponent2 = 0.5 * t * (t - 2) * beta_0 - 0.5 * t ** 2 * beta_1
    term1 = jnp.where(jnp.abs(exponent1) <= 1e-3, -exponent1, 1. - jnp.exp(exponent1))
    term2 = jnp.where(jnp.abs(exponent2) <= 1e-3, -exponent2, 1. - jnp.exp(exponent2))
    return 0.5 * (-2 * jnp.log(term1) + 2 * jnp.log(term2) + beta_0 * (-2 * eps + eps ** 2 - (t - 2) * t) + beta_1 * (-eps ** 2 + t ** 2))

def sample_importance_weighted_time_for_likelihood(shape, beta_0, beta_1, quantile=None, eps=1e-5, steps=100, T=1):
    Z = likelihood_importance_cum_weight(T, beta_0, beta_1, eps)
    if quantile is None:
      quantile = torch.distributions.uniform.Uniform(0,Z.item()).sample((shape,)).numpy()
    lb = jnp.ones_like(quantile) * eps
    ub = jnp.ones_like(quantile) * T

    for i in range(steps):
        mid = (lb + ub) / 2.
        value = likelihood_importance_cum_weight(mid, beta_0, beta_1, eps=eps)
        lb = jnp.where(value <= quantile, mid, lb)
        ub = jnp.where(value <= quantile, ub, mid)
    return (lb + ub) / 2.

# def ce_loss(outputs, targets):
#     loss = nn.BCEWithLogitsLoss()
#     return loss(outputs, targets)

def pos_energy_loss(energy_out):
    log_sigmoid = nn.LogSigmoid()
    return -1*log_sigmoid(-energy_out).mean()

def neg_energy_loss(energy_out):
    log_sigmoid = nn.LogSigmoid()
    return -1*log_sigmoid(energy_out).mean()

def perturb(batch, likelihood_weighting=True, eps=1e-5, T=1, beta_0=0.1, beta_1=20, im_sample=False):

    if likelihood_weighting and im_sample:
        t = torch.tensor(np.array(sample_importance_weighted_time_for_likelihood(batch.shape[0], beta_0, beta_1, T=T))).to(batch.device)
    else:
        t = torch.rand(batch.shape[0], device=batch.device) * (T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = marginal_prob(batch, t, beta_0, beta_1)
    perturbed_data = mean + std.view(-1,*([1]*len(batch.shape[1:]))) * z

    return perturbed_data, t

def train_model(train_loader, image_vae, mask_vae, attr_vae, attr_visible, cl_model, optimizer, device, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, vae_type="VAE", im_sample=False):
    losses = 0
    image_vae.eval()
    mask_vae.eval()
    attr_vae.eval()
    cl_model.train()
    start_time = time.time()

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
                z_image = image_mu
                z_mask =  mask_mu
                z_attr = attr_mu
            elif vae_type == "AE":
                z_image = image_vae.encoder(input)
                z_mask = mask_vae.encoder(masks)
                z_attr = attr_vae.encoder(target.float())

            sel_mod_idx = torch.randperm(3)[:1]
            sel_mod_idx = sel_mod_idx.item()

            if sel_mod_idx == 0:
                z1_pos = z_image
                z2_pos = z_mask
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 0, 1
            elif sel_mod_idx == 1:
                z1_pos = z_image
                z2_pos = z_attr
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 0, 2
            elif sel_mod_idx == 2:
                z1_pos = z_mask
                z2_pos = z_attr
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 1, 2

            # Positive example zs
            z = torch.cat([z1_pos, z2_pos], dim=1).detach()
            # target_pos = torch.ones(z.shape[0],1).to(device)

            z_neg = torch.cat([z1_neg, z2_neg], dim=1).detach()
            # target_neg = torch.zeros(z.shape[0],1).to(device)

            z_neg2 = torch.randn_like(z_neg)
            # target_neg2 = torch.zeros(z.shape[0],1).to(device)

        with torch.enable_grad():
            perturbed_pos, t = perturb(z, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_pos = cl_model(perturbed_pos, t, id1, id2)

            perturbed_neg, t = perturb(z_neg, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_neg = cl_model(perturbed_neg, t, id1, id2)

            cl_out_neg2 = cl_model(z_neg2, t, id1, id2)

            loss_pos = pos_energy_loss(cl_out_pos).mean()
            # loss_neg = ce_loss(cl_out_neg, target_neg).mean()
            loss_neg = (neg_energy_loss(cl_out_neg).mean() + neg_energy_loss(cl_out_neg2).mean()) / 2
            total_loss = loss_pos + loss_neg

            losses += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    end_time = time.time()
    losses /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return losses

def evaluate(test_loader, image_vae, mask_vae, attr_vae, attr_visible, cl_model, device, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, vae_type="VAE", im_sample=False):
    with torch.no_grad():
        start_time = time.time()
        losses = 0
        image_vae.eval()
        mask_vae.eval()
        attr_vae.eval()
        cl_model.eval()

        for batch_idx, (images, masks, target) in enumerate(test_loader):

            input = images.to(device)
            masks = masks.to(device)
            target = target.to(device)[:, attr_visible]

            # Get z for each modality
            if vae_type == "VAE":
                image_mu, image_logvar = image_vae.encoder(input)
                mask_mu, mask_logvar = mask_vae.encoder(masks)
                attr_mu, attr_logvar = attr_vae.encoder(target.float())
                z_image = image_mu
                z_mask =  mask_mu
                z_attr = attr_mu
            elif vae_type == "AE":
                z_image = image_vae.encoder(input)
                z_mask = mask_vae.encoder(masks)
                z_attr = attr_vae.encoder(target.float())

            sel_mod_idx = torch.randperm(3)[:1]
            sel_mod_idx = sel_mod_idx.item()

            if sel_mod_idx == 0:
                z1_pos = z_image
                z2_pos = z_mask
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 0, 1
            elif sel_mod_idx == 1:
                z1_pos = z_image
                z2_pos = z_attr
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 0, 2
            elif sel_mod_idx == 2:
                z1_pos = z_mask
                z2_pos = z_attr
                rand_idx = torch.randperm(z1_pos.shape[0])
                z1_neg = z1_pos[rand_idx]
                z2_neg = z2_pos.clone().detach()
                id1, id2 = 1, 2

            # Positive example zs
            z = torch.cat([z1_pos, z2_pos], dim=1).detach()
            # target_pos = torch.ones(z.shape[0],1).to(device)

            z_neg = torch.cat([z1_neg, z2_neg], dim=1).detach()
            # target_neg = torch.zeros(z.shape[0],1).to(device)

            z_neg2 = torch.randn_like(z_neg)
            # target_neg2 = torch.zeros(z.shape[0],1).to(device)


            perturbed_pos, t = perturb(z, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_pos = cl_model(perturbed_pos, t, id1, id2)

            perturbed_neg, t = perturb(z_neg, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_neg = cl_model(perturbed_neg, t, id1, id2)

            cl_out_neg2 = cl_model(z_neg2, t, id1, id2)

            loss_pos = pos_energy_loss(cl_out_pos).mean()
            # loss_neg = ce_loss(cl_out_neg, target_neg).mean()
            loss_neg = (neg_energy_loss(cl_out_neg).mean() + neg_energy_loss(cl_out_neg2).mean()) / 2
            total_loss = loss_pos + loss_neg

            losses += total_loss.item()     

        losses /= len(test_loader)
        print("VALIDATION TIME TAKEN: ", time.time() - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        return losses
    

def run(epochs, batch_size, lr, size_z1, size_z2, image_model_path, mask_model_path, attr_model_path, savefolder, unq_name, cuda_num, vae_type, beta_0, beta_1, T, likelihood_weighting, im_sample):
    res_size = 128
    print('vars: ', epochs, batch_size, lr, size_z1, size_z2, unq_name, flush=True)
    train_losses, val_losses = [], []
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    print(attr_visible, flush=True)

    likelihood_weighting = True if likelihood_weighting else False
    im_sample = True if im_sample else False

    print("likelihood_weighting: ", likelihood_weighting, " imp: ", im_sample, " T: ", T, " beta0: ", beta_0, " beta1: ", beta_1, " VAE type: ", vae_type, flush=True)

    savefolder += '/'
    save_paths = {'model': './models/' + savefolder, 'plot': './plots/' + savefolder}
    for p in save_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    # cuda = torch.cuda.is_available()
    # print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:" + str(cuda_num))
    print("device: ", str(cuda_num), torch.cuda.get_device_properties(device), flush=True)
    
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

    cl_model = ClwithTime3(n_mod=2, size_z=size_z1, n_class=1)
    optimizer = torch.optim.Adam(cl_model.parameters(), lr=lr)
    cl_model = cl_model.to(device)

    unq_name += '_vtype_' + vae_type + '_b_' + str(beta_0) + '_' + str(beta_1) + '_'
    if likelihood_weighting:
        unq_name += '_ll_'
    if likelihood_weighting and im_sample:
        unq_name += '_ImpSamp_'

    print("unq_name: ", unq_name, flush=True)
    
    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, res_size)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, image_vae, mask_vae, attr_vae, attr_visible, cl_model, optimizer, device, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, vae_type=vae_type, im_sample=im_sample)
        validation_loss = evaluate(val_dataloader, image_vae, mask_vae, attr_vae, attr_visible, cl_model, device, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, vae_type=vae_type, im_sample=im_sample)

        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)

        # if epoch == 0:
        #     prev_loss = validation_loss
        # if epoch > 0 and (validation_loss < prev_loss):
        torch.save({
        'epoch': epoch,
        'model_state_dict': cl_model.state_dict(),
        'train_loss': training_loss,
        'val_loss': validation_loss,
        'size_z': size_z1,
        }, save_paths['model'] + str(size_z1) + str(unq_name))
        print('Model saved', flush=True)
        # prev_loss = validation_loss

        # if (epoch + 1) % 500 == 0:
        #     lr /= 5
        #     sm_optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], save_paths['plot'] + '_' + str(size_z1) + '_' + unq_name)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z1', type=int, default=256,
                        help='size of z1 [default: 256]')
    parser.add_argument('--size-z2', type=int, default=256,
                        help='size of z2 [default: 256]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate [default: 0.0005]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--savefolder', type=str, default='cel_clf_time',
                        help='folder name to save output [default: "cel_clf_time"]')
    parser.add_argument('--unq-name', type=str, default='cel_sde_cls_with_time3_EBM_IND_NEW_',
                        help='identifier name for saving [default: "cel_sde_cls_with_time3_EBM_IND_NEW_"]')
    
    # 256 VAE
    parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__',
                        help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__"]')
    parser.add_argument('--mask-path', type=str, default='./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq',
                        help='mask vae model path [default: "./models/celeba_mask/celeb_hq_mask_dsize_128_z_256_beta_1_mask_celebhq"]')
    parser.add_argument('--attr-path', type=str, default='./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1',
                        help='vae model path [default: "./models/celeba_attr/celeba_attr_bn_hq__z_256_beta_0.1"]')
    
    # 256 AE
    parser.add_argument('--image-path-ae', type=str, default='./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_',
                        help='image path for ae [default: "./models/celeba/celeb_hq_ae__beta_0.0001_noisecons_0.001_z_256_256_re4ne3_sm_hq_"]')
    parser.add_argument('--mask-path-ae', type=str, default='./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_',
                        help='mask path for ae [default: "./models/celeba_mask/celeb_hq_mask_AE__dsize_128_z_256_mask_hq256_re5ne3_"]')
    parser.add_argument('--attr-path-ae', type=str, default='./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1',
                        help='attr path for ae [default: "./models/celeba_attr/celeba_attr_bn_hq_AEreg__z_256_0.0001_att_ae_re4ne1"]')
    
    parser.add_argument('--vae-type', type=str, default='VAE',
                        help='vae type: AE or VAE [default: "VAE"]')
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
    parser.add_argument('--im-sample', type=int, default=1, 
                        help='if 1, use importance sampling for likelihood weighting')
    parser.add_argument('--pc', type=int, default=0, 
                        help='if 1, use langevin corrector')
    parser.add_argument('--n-steps', type=int, default=1, 
                        help='langevin step')
    parser.add_argument('--target-snr', type=float, default=0.16,
                        help='target signal to noise ratio used in langevin step  [default: 0.16]')

    args = parser.parse_args()

    if args.vae_type == "VAE":
        run(args.epochs, args.batch_size, args.lr, args.size_z1, args.size_z2, args.image_path, args.mask_path, args.attr_path, args.savefolder, args.unq_name, args.cuda, \
        args.vae_type, args.beta0, args.beta1, args.T, args.ll_weighting, args.im_sample)
    elif args.vae_type == "AE":
        run(args.epochs, args.batch_size, args.lr, args.size_z1, args.size_z2, args.image_path_ae, args.mask_path_ae, args.attr_path_ae, args.savefolder, args.unq_name, args.cuda, \
        args.vae_type, args.beta0, args.beta1, args.T, args.ll_weighting, args.im_sample)
    else:
        raise Exception("Wrong VAE type")


