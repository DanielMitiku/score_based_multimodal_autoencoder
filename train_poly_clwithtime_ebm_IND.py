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

from polymnist_dataset import get_train_test_dataset_upd10_32x32, test_dataset_upd10_32x32
from h_vae_model_copy import ResVAE, ResAE
from lat_sm2_model import ClwithTime3
from polymnist_model import PMCLF

from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil

from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_32x32()
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

# temporary
# def get_train_test_dataloader_upd10_32x32(batch_size):
#     paired_val_dataset = test_dataset_upd10_32x32(test=False)
#     val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return val_dataloader, val_dataloader, val_dataloader

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


def pos_energy_loss(energy_out):
    log_sigmoid = nn.LogSigmoid()
    return -1*log_sigmoid(-energy_out).mean()

def neg_energy_loss(energy_out):
    log_sigmoid = nn.LogSigmoid()
    return -1*log_sigmoid(energy_out).mean()

# def ce_loss(outputs, targets):
#     loss = nn.BCEWithLogitsLoss()
#     return loss(outputs, targets)

def perturb(batch, likelihood_weighting=True, eps=1e-5, T=1, beta_0=0.1, beta_1=20, im_sample=False):

    if likelihood_weighting and im_sample:
        t = torch.tensor(np.array(sample_importance_weighted_time_for_likelihood(batch.shape[0], beta_0, beta_1, T=T))).to(batch.device)
    else:
        t = torch.rand(batch.shape[0], device=batch.device) * (T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = marginal_prob(batch, t, beta_0, beta_1)
    perturbed_data = mean + std.view(-1,*([1]*len(batch.shape[1:]))) * z

    return perturbed_data, t

def train_model(train_loader, pvae_dict, cl_model, optimizer, device, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, vae_type="VAE", im_sample=False):
    losses = 0
    for mod in sorted(pvae_dict.keys()):
        pvae_dict[mod].eval()
    cl_model.train()
    start_time = time.time()
    start_mod = int(list(pvae_dict.keys())[0])

    for batch_idx, (images, target) in enumerate(train_loader):
        with torch.no_grad():
            p = {}
            z = {}
            z_neg = {}
            sel_mod_idx = torch.randperm(len(pvae_dict))[:2]
            sel_mod_idx = [idx.item() for idx in sel_mod_idx]

            mod1 = str(sel_mod_idx[0] + start_mod)
            p[mod1] = images['m'+mod1].to(device)
            if vae_type == "VAE":
                z1_pos = pvae_dict[mod1].encoder(p[mod1])[0]
            elif vae_type == "AE":
                z1_pos = pvae_dict[mod1].encoder(p[mod1])
            rand_idx = torch.randperm(z1_pos.shape[0])
            z1_neg = z1_pos[rand_idx]

            mod2 = str(sel_mod_idx[1] + start_mod)
            p[mod2] = images['m'+mod2].to(device)
            if vae_type == "VAE":
                z2_pos = pvae_dict[mod2].encoder(p[mod2])[0]
            elif vae_type == "AE":
                z2_pos = pvae_dict[mod2].encoder(p[mod2])
            rand_idx = torch.randperm(z2_pos.shape[0])
            z2_neg = z2_pos[rand_idx]

            # Positive example zs
            z = torch.cat([z1_pos, z2_pos], dim=1).detach()

            z_neg = torch.cat([z1_neg, z2_neg], dim=1).detach()

            z_neg2 = torch.randn_like(z_neg)

        with torch.enable_grad():
            perturbed_pos, t = perturb(z, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_pos = cl_model(perturbed_pos, t, sel_mod_idx[0], sel_mod_idx[1])

            perturbed_neg, t = perturb(z_neg, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_neg = cl_model(perturbed_neg, t, sel_mod_idx[0], sel_mod_idx[1])

            cl_out_neg2 = cl_model(z_neg2, t, sel_mod_idx[0], sel_mod_idx[1])

            loss_pos = pos_energy_loss(cl_out_pos).mean()
            # loss_neg = neg_energy_loss(cl_out_neg).mean()
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


def evaluate(test_loader, pvae_dict, cl_model, device, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, vae_type="VAE", im_sample=False):
    with torch.no_grad():
        start_time = time.time()
        losses = 0
        cl_model.eval()
        start_mod = int(list(pvae_dict.keys())[0])

        for batch_idx, (images, target) in enumerate(test_loader):
            p = {}
            z = {}
            z_neg = {}
            sel_mod_idx = torch.randperm(len(pvae_dict))[:2]
            sel_mod_idx = [idx.item() for idx in sel_mod_idx]

            mod1 = str(sel_mod_idx[0] + start_mod)
            p[mod1] = images['m'+mod1].to(device)
            if vae_type == "VAE":
                z1_pos = pvae_dict[mod1].encoder(p[mod1])[0]
            elif vae_type == "AE":
                z1_pos = pvae_dict[mod1].encoder(p[mod1])
            rand_idx = torch.randperm(z1_pos.shape[0])
            z1_neg = z1_pos[rand_idx]

            mod2 = str(sel_mod_idx[1] + start_mod)
            p[mod2] = images['m'+mod2].to(device)
            if vae_type == "VAE":
                z2_pos = pvae_dict[mod2].encoder(p[mod2])[0]
            elif vae_type == "AE":
                z2_pos = pvae_dict[mod2].encoder(p[mod2])
            rand_idx = torch.randperm(z2_pos.shape[0])
            z2_neg = z2_pos[rand_idx]

            # Positive example zs
            z = torch.cat([z1_pos, z2_pos], dim=1).detach()

            z_neg = torch.cat([z1_neg, z2_neg], dim=1).detach()

            z_neg2 = torch.randn_like(z_neg)

            perturbed_pos, t = perturb(z, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_pos = cl_model(perturbed_pos, t, sel_mod_idx[0], sel_mod_idx[1])

            perturbed_neg, t = perturb(z_neg, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            cl_out_neg = cl_model(perturbed_neg, t, sel_mod_idx[0], sel_mod_idx[1])

            cl_out_neg2 = cl_model(z_neg2, t, sel_mod_idx[0], sel_mod_idx[1])

            loss_pos = pos_energy_loss(cl_out_pos).mean()
            loss_neg = (neg_energy_loss(cl_out_neg).mean() + neg_energy_loss(cl_out_neg2).mean()) / 2
            # loss_neg = neg_energy_loss(cl_out_neg).mean()
            total_loss = loss_pos + loss_neg

            losses += total_loss.item()     

        losses /= len(test_loader)
        print("VALIDATION TIME TAKEN: ", time.time() - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        return losses
    

def run(epochs, batch_size, lr, size_z, all_mods, savefolder, model_paths, unq_name, cuda_num, vae_type, beta_0, beta_1, T, likelihood_weighting, im_sample):
    print('vars: ', epochs, batch_size, lr, size_z, unq_name, all_mods, savefolder, flush=True)
    train_losses, val_losses = [], []

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
    
    enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
    dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
    size_in = 32
    img_ch = 3
    pvae_dict = {}
    n_mod = len(all_mods)
    
    for ind, model_path in enumerate(model_paths):
        if str(ind) in all_mods:
            if vae_type == "VAE":
                pmvae = ResVAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
            elif vae_type == "AE":
                pmvae = ResAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
            else:
                raise Exception("wrong vae type")
            pmvae.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
            pmvae = pmvae.to(device)
            pvae_dict[str(ind)] = pmvae

    cl_model = ClwithTime3(n_mod=2, size_z=size_z, n_class=1)
    optimizer = torch.optim.Adam(cl_model.parameters(), lr=lr)
    cl_model = cl_model.to(device)

    unq_name += '_vtype_' + vae_type + '_b_' + str(beta_0) + '_' + str(beta_1) + '_'
    if likelihood_weighting:
        unq_name += '_ll_'
    if likelihood_weighting and im_sample:
        unq_name += '_ImpSamp_'

    print("unq_name: ", unq_name, flush=True)
    
    train_dataloader, val_dataloader, _ = get_train_test_dataloader_upd10_32x32(batch_size)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, pvae_dict, cl_model, optimizer, device, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, vae_type=vae_type, im_sample=im_sample)
        validation_loss = evaluate(val_dataloader, pvae_dict, cl_model, device, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, vae_type=vae_type, im_sample=im_sample)

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
        'size_z': size_z,
        }, save_paths['model'] + all_mods + '_' + str(size_z) + str(unq_name))
        print('Model saved', flush=True)
        # prev_loss = validation_loss

        # if (epoch + 1) % 500 == 0:
        #     lr /= 5
        #     sm_optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], save_paths['plot'] + all_mods + '_' + str(size_z) + '_' + unq_name)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train [default: 3000]')
    parser.add_argument('--upd', type=str, default='',
                        help='updated polymnist dataset [default: ]')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate [default: 0.0005]')
    parser.add_argument('--sm-type', type=str, default='sde',
                        help='loss type [default: "sde"]')
    parser.add_argument('--allmods', type=str, default='01',
                        help='Mods to train on [default: "01"]')
    parser.add_argument('--savefolder', type=str, default='poly_clf_time',
                        help='folder name to save output [default: "poly_clf_time"]')
    parser.add_argument('--unq-name', type=str, default='sde_cls_with_time_EBM_IND_',
                        help='identifier name for saving [default: "sde_cls_with_time_EBM_IND_"]')
    parser.add_argument('--p0-path', type=str, default='./models/polyupd10_m0/polyupd10_m0_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m0/polyupd10_m0_res_beta_0.1__64"]')
    parser.add_argument('--p1-path', type=str, default='./models/polyupd10_m1/polyupd10_m1_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m1/polyupd10_m1_res_beta_0.1__64"]')
    parser.add_argument('--p2-path', type=str, default='./models/polyupd10_m2/polyupd10_m2_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m2/polyupd10_m2_res_beta_0.1__64"]')
    parser.add_argument('--p3-path', type=str, default='./models/polyupd10_m3/polyupd10_m3_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m3/polyupd10_m3_res_beta_0.1__64"]')
    parser.add_argument('--p4-path', type=str, default='./models/polyupd10_m4/polyupd10_m4_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m4/polyupd10_m4_res_beta_0.1__64"]')
    parser.add_argument('--p5-path', type=str, default='./models/polyupd10_m5/polyupd10_m5_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m5/polyupd10_m5_res_beta_0.1__64"]')
    parser.add_argument('--p6-path', type=str, default='./models/polyupd10_m6/polyupd10_m6_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m6/polyupd10_m6_res_beta_0.1__64"]')
    parser.add_argument('--p7-path', type=str, default='./models/polyupd10_m7/polyupd10_m7_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m7/polyupd10_m7_res_beta_0.1__64"]')
    parser.add_argument('--p8-path', type=str, default='./models/polyupd10_m8/polyupd10_m8_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m8/polyupd10_m8_res_beta_0.1__64"]')
    parser.add_argument('--p9-path', type=str, default='./models/polyupd10_m9/polyupd10_m9_res_beta_0.1__64',
                        help='multimodal model path [default: "./models/polyupd10_m9/polyupd10_m9_res_beta_0.1__64"]')
    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')
    
    parser.add_argument('--p0-path-ae', type=str, default='./models/polyupd10_m0/polyNEWAE_m0_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m0/polyNEWAE_m0_64_0.01_1e-05"]')
    parser.add_argument('--p1-path-ae', type=str, default='./models/polyupd10_m1/polyNEWAE_m1_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m1/polyNEWAE_m1_64_0.01_1e-05"]')
    parser.add_argument('--p2-path-ae', type=str, default='./models/polyupd10_m2/polyNEWAE_m2_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m2/polyNEWAE_m2_64_0.01_1e-05"]')
    parser.add_argument('--p3-path-ae', type=str, default='./models/polyupd10_m3/polyNEWAE_m3_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m3/polyNEWAE_m3_64_0.01_1e-05"]')
    parser.add_argument('--p4-path-ae', type=str, default='./models/polyupd10_m4/polyNEWAE_m4_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m4/polyNEWAE_m4_64_0.01_1e-05"]')
    parser.add_argument('--p5-path-ae', type=str, default='./models/polyupd10_m5/polyNEWAE_m5_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m5/polyNEWAE_m5_64_0.01_1e-05"]')
    parser.add_argument('--p6-path-ae', type=str, default='./models/polyupd10_m6/polyNEWAE_m6_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m6/polyNEWAE_m6_64_0.01_1e-05"]')
    parser.add_argument('--p7-path-ae', type=str, default='./models/polyupd10_m7/polyNEWAE_m7_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m7/polyNEWAE_m7_64_0.01_1e-05"]')
    parser.add_argument('--p8-path-ae', type=str, default='./models/polyupd10_m8/polyNEWAE_m8_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m8/polyNEWAE_m8_64_0.01_1e-05"]')
    parser.add_argument('--p9-path-ae', type=str, default='./models/polyupd10_m9/polyNEWAE_m9_64_0.01_1e-05',
                        help='multimodal model path-ae [default: "./models/polyupd10_m9/polyNEWAE_m9_64_0.01_1e-05"]')

    
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
        vae_paths = [args.p0_path, args.p1_path, args.p2_path, args.p3_path, args.p4_path, args.p5_path, args.p6_path, args.p7_path, args.p8_path, args.p9_path]
    elif args.vae_type == "AE":
        vae_paths = [args.p0_path_ae, args.p1_path_ae, args.p2_path_ae, args.p3_path_ae, args.p4_path_ae, args.p5_path_ae, args.p6_path_ae, args.p7_path_ae, args.p8_path_ae, args.p9_path_ae]
    else:
        raise Exception("Wrong VAE type")
    
    run(args.epochs, args.batch_size, args.lr, args.size_z, args.allmods, args.savefolder, vae_paths, args.unq_name, args.cuda, \
        args.vae_type, args.beta0, args.beta1, args.T, args.ll_weighting, args.im_sample)


