import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"

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
from lat_sm2_model import ClwithTime2, ClwithTime3
from unet_model import Unet
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
def get_train_test_dataloader_upd10_32x32_val(batch_size):
    paired_val_dataset = test_dataset_upd10_32x32(test=False)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return val_dataloader, val_dataloader, val_dataloader

def ce_loss(outputs, targets, cl_g):
    if cl_g.n_class == 1:
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)

# Taken and updated from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
def sde(x, t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

def rev_sde(x, t, score_fn, beta_0, beta_1, probability_flow, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = sde(x, t, beta_0, beta_1)
    score = score_fn(x, t)

    ## if classifier guidance
    # if cl_g is not None:
    #     with torch.enable_grad():
    #         x.requires_grad = True
    #         cl_out = cl_g(x.view(x.shape[0],-1), t)
    #         # cl_loss = ce_loss(cl_out, target, cl_g)
    #         # grad = torch.autograd.grad(cl_loss, x)[0]
    #         grad = torch.autograd.grad(cl_out.mean(), x)[0]
    #         if cl_s is not None:
    #             # score += cl_s * grad
    #             score -= cl_s * grad
    #         else:
    #             auto_scale = torch.norm(score) / torch.norm(grad)
    #             # score += grad
    #             score -= auto_scale * grad

    if cl_g is not None and len(given) > 0:
        with torch.enable_grad():
            mod1_idx = torch.randint(len(given),(1,)).item()
            mod1 = given[mod1_idx]
            predicted = ''.join([m for m in all_mods if m not in given])
            mod2_idx = torch.randint(len(predicted),(1,)).item()
            mod2 = predicted[mod2_idx]
            new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
            new_x.requires_grad = True
            # cl_out = cl_g(new_x.view(new_x.shape[0],-1), t)
            cl_out = cl_g(new_x.view(new_x.shape[0],-1), t, int(mod1)-int(all_mods[0]), int(mod2)-int(all_mods[0]))
            # cl_loss = ce_loss(cl_out, target, cl_g)
            # grad = torch.autograd.grad(cl_loss, new_x)[0]
            grad = torch.autograd.grad(cl_out.mean(), new_x)[0]
            if cl_s is not None:
                # score[:,int(mod1)-int(all_mods[0])] -= cl_s * grad[:,0]
                score[:,int(mod2)-int(all_mods[0])] -= cl_s * grad[:,1]
    
    drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
    # Set the diffusion function to zero for ODEs.
    diffusion = 0. if probability_flow else diffusion
    return drift, diffusion

def em_predictor(x, t, score_fn, beta_0=0.1, beta_1=20, N=1000, probability_flow=False, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
    dt = -1. / N
    z = torch.randn_like(x)
    drift, diffusion = rev_sde(x, t, score_fn, beta_0, beta_1, probability_flow,cl_g, cl_s, target, given=given, all_mods=all_mods)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean

def corrector(x, t, score_fn, beta_0, beta_1, N, T, n_steps, target_snr, cl_g=None, cl_s=None, target=None, given=None, all_mods=None):
    discrete_betas = torch.linspace(beta_0 / N, beta_1 / N, N)
    alphas = 1. - discrete_betas
    timestep = (t * (N - 1) / T).long()
    alpha = alphas.to(t.device)[timestep]

    for i in range(n_steps):
        grad = score_fn(x, t)

        # # if classifier guidance
        # if cl_g is not None:
        #     with torch.enable_grad():
        #         x.requires_grad = True
        #         cl_out = cl_g(x.view(x.shape[0],-1), t)
        #         # cl_loss = ce_loss(cl_out, target, cl_g)
        #         # cls_grad = torch.autograd.grad(cl_loss, x)[0]
        #         cls_grad = torch.autograd.grad(cl_out.mean(), x)[0]
        #         if cl_s is not None:
        #             # grad += cl_s * cls_grad
        #             grad -= cl_s * cls_grad
        #         else:
        #             auto_scale = torch.norm(grad) / torch.norm(cls_grad)
        #             # grad += cls_grad
        #             grad -= auto_scale * cls_grad

        if cl_g is not None and len(given) > 0:
            with torch.enable_grad():
                mod1_idx = torch.randint(len(given),(1,)).item()
                mod1 = given[mod1_idx]
                predicted = ''.join([m for m in all_mods if m not in given])
                mod2_idx = torch.randint(len(predicted),(1,)).item()
                mod2 = predicted[mod2_idx]
                new_x = torch.cat([x[:,int(mod1)-int(all_mods[0]),:,:].unsqueeze(1), x[:,int(mod2)-int(all_mods[0]),:,:].unsqueeze(1)], dim=1)
                new_x.requires_grad = True
                # cl_out = cl_g(new_x.view(new_x.shape[0],-1), t)
                cl_out = cl_g(new_x.view(new_x.shape[0],-1), t, int(mod1)-int(all_mods[0]), int(mod2)-int(all_mods[0]))
                # cl_loss = ce_loss(cl_out, target, cl_g)
                # cls_grad = torch.autograd.grad(cl_loss, new_x)[0]
                cls_grad = torch.autograd.grad(cl_out.mean(), new_x)[0]

                if cl_s is not None:
                    # grad[:,int(mod1)-int(all_mods[0])] -= cl_s * cls_grad[:,0]
                    grad[:,int(mod2)-int(all_mods[0])] -= cl_s * cls_grad[:,1]

                # x.requires_grad = True
                # new_x = x[:,:2,:,:]
                # cl_out = cl_g(new_x.view(x.shape[0],-1), t)
                # cl_loss = ce_loss(cl_out, target, cl_g)
                # cls_grad = torch.autograd.grad(cl_loss, new_x)[0]

                # if cl_s is not None:
                #     # grad += cl_s * cls_grad
                #     grad[:,1] -= cl_s * cls_grad[:,1]

        noise = torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean

def marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    # print('marginal prob x.shape: ',x.shape, 't: ', t.shape, 'logmean: ', log_mean_coeff[:, None, None, None].shape, flush=True)
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def uncond_sampler(sample_shape, model, device, T=1, eps=1e-3, beta_0=0.1, beta_1=20, N=1000, probability_flow=False, pc=False, n_steps=1, target_snr=0.16, cl_g=None, cl_s=None, target=None):
    with torch.no_grad():
        # Initial sample
        x = torch.randn(sample_shape).to(device)
        timesteps = torch.linspace(T, eps, N, device=device)

        for i in range(N):
            t = timesteps[i]
            vec_t = torch.ones(sample_shape[0], device=t.device) * t
            x, x_mean = em_predictor(x, vec_t, model, beta_0, beta_1, N, probability_flow, cl_g, cl_s, target)

            if pc:
                x, x_mean = corrector(x, vec_t, model, beta_0, beta_1, N, T, n_steps, target_snr, cl_g, cl_s, target)

        return x_mean


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

def loss_fn(batch, score_fn, reduce_mean=True, likelihood_weighting=True, eps=1e-5, T=1, beta_0=0.1, beta_1=20, im_sample=False):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    if likelihood_weighting and im_sample:
        t = torch.tensor(np.array(sample_importance_weighted_time_for_likelihood(batch.shape[0], beta_0, beta_1, T=T))).to(batch.device)
    else:
        t = torch.rand(batch.shape[0], device=batch.device) * (T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = marginal_prob(batch, t, beta_0, beta_1)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
        if im_sample:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde(torch.zeros_like(batch), t, beta_0, beta_1)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss


def train_model(train_loader, pvae_dict, sm_model, optimizer, sm_type, device, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, vae_type="VAE", im_sample=False):
    losses = 0
    for mod in sorted(pvae_dict.keys()):
        pvae_dict[mod].eval()
    sm_model.train()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):
        with torch.no_grad():
            p = {}
            z = {}
            for mod in pvae_dict.keys():
                p[mod] = images['m'+mod].to(device)
                if vae_type == "VAE":
                    z[mod] = pvae_dict[mod].encoder(p[mod])[0]
                elif vae_type == "AE":
                    z[mod] = pvae_dict[mod].encoder(p[mod])

            # stack zs
            z_all = torch.cat([z[mod].unsqueeze(1) for mod in sorted(pvae_dict.keys())], dim=1).view(-1,len(pvae_dict.keys()),8,8).detach()

        with torch.enable_grad():
            loss = loss_fn(z_all, sm_model, reduce_mean=True, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            losses += loss.item()        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    losses /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return losses


def evaluate(test_loader, pvae_dict, sm_model, amount, size_z, sm_type, epoch, device, unq_name, save_paths, all_mods, likelihood_weighting=True, T=1, beta_0=0.1, beta_1=20, N=1000, eps=1e-3, noise_obs=False, vae_type="VAE", pc=False, n_steps=1, target_snr=0.16, im_sample=False, cl_g=None, cl_s=None):
    with torch.no_grad():
        start_time = time.time()
        losses = 0
        sm_model.eval()
        z = {}
        img_outs = {}
        outs = {}
        
        for mod in sorted(pvae_dict.keys()):
            pvae_dict[mod].eval()
            z[mod] = torch.normal(mean=0, std=1, size=(amount,size_z), requires_grad=True, device=device)
            img_outs[mod] = [] 

        # unconditional sampling
        if (epoch + 1) % 10 == 0:
            z_out = uncond_sampler((amount,len(all_mods),8,8), sm_model, device, T=T, eps=eps, beta_0=beta_0, beta_1=beta_1, N=N, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=None, cl_s=None, target=None)
            for ind,mod in enumerate(sorted(pvae_dict.keys())):
                outs[mod] = pvae_dict[mod].decoder(z_out[:,ind,:, :].view(amount,size_z))
                img_outs[mod].append(torch.clip(outs[mod],0.,1.).squeeze().permute(1,2,0))

            for mod in sorted(pvae_dict.keys()):
                img_outs[mod] = torch.cat(img_outs[mod], dim=0)
            img_outs = torch.cat([img_outs[mod] for mod in sorted(pvae_dict.keys())], dim=1)
            plt.figure(figsize=(10,10))
            plt.imshow(img_outs.cpu().numpy())
            plt.axis('off')
            plt.savefig(save_paths['image'] +  all_mods + '_' + sm_type + unq_name + '__' + str(epoch) + '.png')


        for batch_idx, (images, target) in enumerate(test_loader):
            p = {}
            z = {}
            for mod in sorted(pvae_dict.keys()):
                p[mod] = images['m'+mod].to(device)
                if vae_type == "VAE":
                    z[mod] = pvae_dict[mod].encoder(p[mod])[0]
                elif vae_type == "AE":
                    z[mod] = pvae_dict[mod].encoder(p[mod])
            target = target.to(device)

            # stack zs
            z_all = torch.cat([z[mod].unsqueeze(1) for mod in sorted(pvae_dict.keys())], dim=1).view(-1,len(pvae_dict.keys()),8,8).detach()

            loss = loss_fn(z_all, sm_model, reduce_mean=True, likelihood_weighting=likelihood_weighting, eps=1e-5, T=T, beta_0=beta_0, beta_1=beta_1, im_sample=im_sample)
            losses += loss.item()

        # conditional sampling
        img_outs = {}
        given = all_mods[0]
        noised = {}
        outs = {}

        if (epoch + 1) % 10 == 0:
            for mod in sorted(pvae_dict.keys()):
                if mod not in given:
                    z[mod] = torch.normal(mean=0, std=1, size=(amount,size_z), requires_grad=True, device=device)
                else:
                    z[mod] = z[mod][0].view(1,-1)
                img_outs[mod] = [] 

            if cl_g is not None:
                if cl_g.n_class == 1:
                    clg_target = torch.ones(z[all_mods[0]].shape[0], 1).to(device)
                else:
                    clg_target = target
            else:
                clg_target = target
        
            timesteps = torch.linspace(T, eps, N, device=device)

            for i in range(N):
                t = timesteps[i]
                vec_t = torch.ones(z[all_mods[0]].shape[0], device=t.device) * t

                for mod in all_mods:
                    if noise_obs:
                        if mod in given:
                            mean, std = marginal_prob(z[mod].view(-1,1,8,8), vec_t, beta_0, beta_1)
                            noised[mod] = (mean + std[:, None, None, None] * z[mod].view(-1,1,8,8)).view(-1, size_z)
                        else:
                            noised[mod] = z[mod]
                    else:
                        noised[mod] = z[mod]

                z_upd = torch.cat([noised[mod].unsqueeze(1) for mod in sorted(pvae_dict.keys())], dim=1).view(-1,len(pvae_dict.keys()),8,8).detach()
                z_upd, z_mean = em_predictor(z_upd, vec_t, sm_model, beta_0, beta_1, N, cl_g=cl_g, cl_s=cl_s, target=clg_target, given=given, all_mods=all_mods)
                if pc:
                    z_upd, z_mean = corrector(z_upd, vec_t, sm_model, beta_0, beta_1, N, T, n_steps, target_snr, cl_g=cl_g, cl_s=cl_s, target=clg_target, given=given, all_mods=all_mods)
                    

                for ind,mod in enumerate(sorted(pvae_dict.keys())):
                    if mod not in given:
                        z[mod] =  z_upd[:,ind].view(amount,size_z)
  
            
            for ind,mod in enumerate(sorted(pvae_dict.keys())):
                if mod not in given:
                    z[mod] =  z_mean[:,ind].view(amount,size_z)
                
            for ind,mod in enumerate(sorted(pvae_dict.keys())):
                outs[mod] = pvae_dict[mod].decoder(z[mod])
                img_outs[mod].append(torch.clip(outs[mod],0.,1.).squeeze().permute(1,2,0))

            for mod in sorted(pvae_dict.keys()):
                img_outs[mod] = torch.cat(img_outs[mod], dim=0)
            img_outs = torch.cat([img_outs[mod] for mod in sorted(pvae_dict.keys())], dim=1)
            plt.figure(figsize=(10,10))
            plt.imshow(img_outs.cpu().numpy())
            plt.axis('off')
            plt.savefig(save_paths['image'] + '_given_' + given + '_' +all_mods + '_' + sm_type + unq_name + '__' + str(epoch) + '.png')
            plt.close('all') 

        losses /= len(test_loader)
        print("VALIDATION TIME TAKEN: ", time.time() - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        return losses   
    

def calc_poly_cond(test_loader, sample_path, sm_model, pvae_dict, mix_type, predicted_mods, all_mods, p_clf, size_z, device, write_input=True, use_mean=True, T=1, beta_0=0.1, beta_1=20, N=1000, eps=1e-3, noise_obs=False, vae_type="VAE", pc=False, n_steps=1, target_snr=0.16, cl_g=None, cl_s=None):
    with torch.no_grad():
        given = ''.join([m for m in all_mods if m not in predicted_mods]) # observed mods
        for vae in pvae_dict.values():
            vae.eval()
        sm_model.eval()
        cond_accuracies = {}
        for pred in predicted_mods:
            cond_accuracies[pred] = 0
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in all_mods:
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            z = {}
            noised = {}

            for key in sorted(pvae_dict.keys()):
                if key in predicted_mods:
                    z[key] = torch.normal(mean=0, std=1, size=(p[key].shape[0],size_z), requires_grad=True, device=device)
                else:
                    if vae_type == "AE":
                        z[key] = pvae_dict[key].encoder(p[key])
                    elif vae_type == "VAE":
                        if use_mean:
                            z[key] = pvae_dict[key].encoder(p[key])[0]
                        else:
                            z[key] = pvae_dict[key].reparametrize(*pvae_dict[key].encoder(p[key])) 
            
            if cl_g is not None:
                if cl_g.n_class == 1:
                    clg_target = torch.ones(z[all_mods[0]].shape[0], 1).to(device)
                else:
                    clg_target = target
            else:
                clg_target = target

            timesteps = torch.linspace(T, eps, N, device=device)

            for i in range(N):
                t = timesteps[i]
                vec_t = torch.ones(z[all_mods[0]].shape[0], device=t.device) * t

                for mod in all_mods:
                    if noise_obs:
                        if mod not in predicted_mods:
                            mean, std = marginal_prob(z[mod].view(-1,1,8,8), vec_t, beta_0, beta_1)
                            noised[mod] = (mean + std[:, None, None, None] * z[mod].view(-1,1,8,8)).view(-1, size_z)
                        else:
                            noised[mod] = z[mod]
                    else:
                        noised[mod] = z[mod]

                z_upd = torch.cat([noised[mod].unsqueeze(1) for mod in sorted(pvae_dict.keys())], dim=1).view(-1,len(pvae_dict.keys()),8,8).detach()
                z_upd, z_mean = em_predictor(z_upd, vec_t, sm_model, beta_0, beta_1, N, cl_g=cl_g, cl_s=cl_s, target=clg_target, given=given, all_mods=all_mods)
                if pc:
                    z_upd, z_mean = corrector(z_upd, vec_t, sm_model, beta_0, beta_1, N, T, n_steps, target_snr, cl_g=cl_g, cl_s=cl_s, target=clg_target, given=given, all_mods=all_mods)

                for ind,mod in enumerate(sorted(pvae_dict.keys())):
                    if mod in predicted_mods:
                        z[mod] =  z_upd[:,ind].view(-1,size_z)

            for ind,mod in enumerate(sorted(pvae_dict.keys())):
                if mod in predicted_mods:
                    z[mod] =  z_mean[:,ind].view(-1,size_z)

            for mod in predicted_mods:
                p_out[mod] = pvae_dict[mod].decoder(z[mod])
                predicted_out = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
                predicted_out = torch.argmax(predicted_out, 1)
                
                cond_acc = torch.sum(predicted_out == target).item()
                cond_acc = cond_acc / p[mod].shape[0]
                cond_accuracies[mod] += cond_acc
                
                if write_input:
                    save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + mod + '_' + ''.join([i for i in all_mods if i not in predicted_mods])] + str(batch_idx) + '_')

        avg_cond = 0
        for mod in cond_accuracies.keys():
            cond_accuracies[mod] /= len(test_loader)
            avg_cond += cond_accuracies[mod]
        print("Cond Coherence: ", cond_accuracies, flush=True)
        print("AVG cond coherence: ", avg_cond / len(cond_accuracies), flush=True)

        fid_scores_cond = {}
        for pred in predicted_mods:
            fid_p = calculate_fid_given_paths([sample_path['p'+pred], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mods])]], 256, device, 2048, 2)
            fid_scores_cond[pred] = fid_p

        return fid_scores_cond, cond_accuracies


def run(epochs, batch_size, lr, size_z, all_mods, savefolder, model_paths, sm_type, unq_name, cuda_num, pclf_path, vae_type, beta_0, beta_1, N, T, likelihood_weighting, noise_obs, pc, n_steps, target_snr, im_sample, use_clg, clg_path, cl_s, eval_only, score_path):
    print('\n vars: ', epochs, batch_size, lr, size_z, sm_type, unq_name, all_mods, savefolder, flush=True)
    train_losses, val_losses = [], []
    eps=1e-3

    # Main vars
    # likelihood_weighting=True
    # likelihood_weighting=False
    # T=1
    # beta_0=0.1
    # # beta_0 = 0.0001
    # # beta_1=20
    # beta_1 = 1
    # # beta_1 = 0.02
    # # N=1000
    # N=100
    # noise_obs = True
    # # noise_obs = False

    likelihood_weighting = True if likelihood_weighting else False
    noise_obs = True if noise_obs else False
    im_sample = True if im_sample else False
    pc = True if pc else False
    use_clg = True if use_clg else False
    eval_only = True if eval_only else False

    # if cl_s == 0:
    #     cl_s = None

    print("likelihood_weighting: ", likelihood_weighting, " imp: ", im_sample, " T: ", T, " beta0: ", beta_0, " beta1: ", beta_1, " N: ", N, " noise_obs: ", noise_obs, " VAE type: ", vae_type, " pc: ", pc,  " n-step: ", n_steps, " snr: ", target_snr, flush=True)
    print("use clg: ", use_clg, " cl_scale: ", cl_s, flush=True)
    if eval_only:
        print("Eval only: ", eval_only, " score path: ", score_path, flush=True)
        if use_clg:
            print("Clg path: ", clg_path, flush=True)

    savefolder += '/'
    save_paths = {'model': './models/' + savefolder, 'plot': './plots/' + savefolder, 'image': './images/' + savefolder}
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

    if n_mod > 5:
        dim = 128
    else:
        dim = 64
    # dim = 128
    score_model = Unet(dim=dim, channels=n_mod, dim_mults=(1,2,2,2), with_time_emb=True)
    # score_model = Unet(dim=dim, channels=n_mod, dim_mults=(1,2,4,8), with_time_emb=True, use_convnext=False)

    if not eval_only:
        sm_optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
        score_model = score_model.to(device)
    else:
        score_model.load_state_dict(torch.load(score_path, map_location=device)['model_state_dict'])
        score_model = score_model.to(device)
        score_model.eval()

    if use_clg:
        # cl_model = ClwithTime(n_mod=n_mod, size_z=size_z, n_class=10)
        # cl_model = ClwithTime2(n_mod=n_mod, size_z=size_z, n_class=1)
        # cl_model = ClwithTime2(n_mod=2, size_z=size_z, n_class=1)
        cl_model = ClwithTime3(n_mod=2, size_z=size_z, n_class=1)
        cl_model.load_state_dict(torch.load(clg_path, map_location=device)['model_state_dict'])
        cl_model = cl_model.to(device)
        cl_model.eval()
    else:
        cl_model = None

    unq_name += '_vtype_' + vae_type + '_dim_' + str(dim) + '_N_' + str(N) + '_b_' + str(beta_0) + '_' + str(beta_1) + '_'
    if likelihood_weighting:
        unq_name += '_ll_'
    if likelihood_weighting and im_sample:
        unq_name += '_ImpSamp_'
    if noise_obs:
        unq_name += '_n_obs_'
    if pc:
        unq_name += '_pc_' + str(pc) + '_snr_' + str(target_snr) + '_'

    print("unq_name: ", unq_name, flush=True)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path, map_location=device))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()
    
    if eval_only:
        _, val_dataloader, _ = get_train_test_dataloader_upd10_32x32_val(batch_size)
    else:
        train_dataloader, val_dataloader, _ = get_train_test_dataloader_upd10_32x32(batch_size)

    best_average_fid = 1000
    best_epoch = 0

    sample_path = {}
    temp_rand = torch.randint(1000000, size=(1,)).item()
    mix_type = sm_type
    temp_dir_name = './t_' + mix_type + str(unq_name) + all_mods + '_' + str(temp_rand) + '/'
    print('temp dir: ', temp_dir_name, flush=True)

    sample_input_path =[]
    sample_output_path = []

    for mod in all_mods:
        sample_path['p' + mod] = temp_dir_name + 'p' + mod + '/'
        sample_input_path.append(temp_dir_name + 'p' + mod + '/')

        predicted_mod = mod
        pred = mod
        sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod])] \
            = temp_dir_name + 'cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod]) + '/'
        sample_output_path.append(temp_dir_name + 'cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod]) + '/')

    if len(all_mods) > 2:
        for i in range(1,len(all_mods)):
            predicted_mod = all_mods[i:]
            for pred in predicted_mod:
                sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod])] \
                    = temp_dir_name + 'cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod]) + '/'
                sample_output_path.append(temp_dir_name + 'cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mod]) + '/')

    for p in sample_input_path + sample_output_path:
        if not os.path.exists(p):
            os.makedirs(p)

    if not eval_only:

        for epoch in range(epochs):
            print("Epoch: "+str(epoch + 1), flush=True)

            training_loss = train_model(train_dataloader, pvae_dict, score_model, sm_optimizer, sm_type, device, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, vae_type=vae_type, im_sample=im_sample)
            validation_loss = evaluate(val_dataloader, pvae_dict, score_model, 1, size_z, sm_type, epoch, device, unq_name, save_paths, all_mods, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, im_sample=im_sample, cl_g=cl_model, cl_s=cl_s)

            print(' ', flush=True)

            train_losses.append(training_loss)
            val_losses.append(validation_loss)

            if epoch == 0:
                prev_loss = validation_loss
            if epoch > 0 and (validation_loss < prev_loss):
                torch.save({
                'epoch': epoch,
                'model_state_dict': score_model.state_dict(),
                # 'optimizer_state_dict': sm_optimizer.state_dict(),
                'train_loss': training_loss,
                'val_loss': validation_loss,
                'size_z': size_z,
                }, save_paths['model'] + all_mods + '_' + str(size_z) + '_VAE_' + str(unq_name))
                print('Model saved', flush=True)
                prev_loss = validation_loss

            # if (epoch + 1) % 500 == 0:
            #     lr /= 5
            #     sm_optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

            # if (epoch + 1) == epochs:
            torch.save({
            'epoch': epoch,
            'model_state_dict': score_model.state_dict(),
            # 'optimizer_state_dict': sm_optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'size_z': size_z,
            }, save_paths['model'] + all_mods + '_' + str(size_z) + '_VAE_' + str(unq_name) + '_last_epoch')
            print('Last Model saved', flush=True)

            if (epoch + 1) % 500 == 0:
                # calculate average cond FID
                average_fid = 0
                average_acc = 0 
                for mod in all_mods:
                    fid_score, cond_acc = calc_poly_cond(val_dataloader, sample_path, score_model, pvae_dict, mix_type, mod, all_mods, poly_clf, size_z, device, write_input=True, use_mean=True, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s)
                    print("fid ", mod, fid_score[mod], flush=True)
                    average_fid += fid_score[mod]
                    average_acc += cond_acc[mod]
                
                average_fid /= len(all_mods)
                average_acc /= len(all_mods)
                print("Epoch: ", epoch+1, " Average cond-FID: ", average_fid, flush=True)
                print("Epoch: ", epoch+1, " Average cond-ACC: ", average_acc, flush=True)

                if average_fid < best_average_fid:
                    best_average_fid = average_fid
                    best_epoch = epoch + 1

                if len(all_mods) > 2:
                    for i in range(1,len(all_mods)):
                        predicted_mods = all_mods[i:]
                        average_fid = 0 
                        fid_score, _ = calc_poly_cond(val_dataloader, sample_path, score_model, pvae_dict, mix_type, predicted_mods, all_mods, poly_clf, size_z, device, write_input=True, use_mean=True, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s)
                        print("fids ", fid_score, flush=True)
                        break
                        
                        for mod in predicted_mods:
                            average_fid += fid_score[mod]
                        average_fid /= len(fid_score)
                        print(" Average cond-FID given ", all_mods[0:i], " ", average_fid, flush=True)

                print(" ")


        print("best average fid: ", best_average_fid, ' at epoch: ', best_epoch, flush=True)
    
        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], save_paths['plot'] + all_mods + '_' + str(size_z) + sm_type + '_' + unq_name)
    
    else:
        with torch.no_grad():
            validation_loss = evaluate(val_dataloader, pvae_dict, score_model, 1, size_z, sm_type, 1, device, unq_name, save_paths, all_mods, likelihood_weighting=likelihood_weighting, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, im_sample=im_sample, cl_g=cl_model, cl_s=cl_s)
            if len(all_mods) <= 2:
                for cl_s in [0, 1, 10, 100, 1000, 10000, 50000, 100000, 1000000]:
                    print(" cl s: ", cl_s, flush=True)
                    average_fid = 0
                    for mod in all_mods:
                        fid_score, _ = calc_poly_cond(val_dataloader, sample_path, score_model, pvae_dict, mix_type, mod, all_mods, poly_clf, size_z, device, write_input=True, use_mean=True, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s)
                        print("fid ", mod, fid_score[mod], flush=True)
                        average_fid += fid_score[mod]
                    
                    average_fid /= len(all_mods)
                    print(" Average cond-FID: ", average_fid, flush=True)
                    print(" ", flush=True)

            if len(all_mods) > 2:
                # for cl_s in [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 40, 50, 80, 100, 1000]: # , 1000, 10000, 100000, 1000000]:
                print(" cl s: ", cl_s, flush=True)
                for i in range(1,len(all_mods)):
                    predicted_mods = all_mods[i:]
                    average_fid = 0 
                    fid_score, _ = calc_poly_cond(val_dataloader, sample_path, score_model, pvae_dict, mix_type, predicted_mods, all_mods, poly_clf, size_z, device, write_input=True, use_mean=True, T=T, beta_0=beta_0, beta_1=beta_1, N=N, eps=eps, noise_obs=noise_obs, vae_type=vae_type, pc=pc, n_steps=n_steps, target_snr=target_snr, cl_g=cl_model, cl_s=cl_s)
                    print("fids ", fid_score, flush=True)
                    
                    for mod in predicted_mods:
                        average_fid += fid_score[mod]
                    average_fid /= len(fid_score)
                    print(" Average cond-FID given ", all_mods[0:i], " ", average_fid, flush=True)
                    print(" ", flush=True)
                    break

    shutil.rmtree(temp_dir_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--upd', type=str, default='',
                        help='updated polymnist dataset [default: ]')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate [default: 0.0005]')
    parser.add_argument('--sm-type', type=str, default='sde',
                        help='loss type [default: "sde"]')
    parser.add_argument('--allmods', type=str, default='01',
                        help='Mods to train on [default: "01"]')
    parser.add_argument('--savefolder', type=str, default='sde_poly',
                        help='folder name to save output [default: "sde_poly"]')
    parser.add_argument('--unq-name', type=str, default='sde_unet_',
                        help='identifier name for saving [default: "sde_unet_"]')
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
    parser.add_argument('--beta1', type=float, default=5,
                        help='beta1  [default: 5]')
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
    
    parser.add_argument('--use-clg', type=int, default=0, 
                        help='if 1, use classifier guidance')
    # parser.add_argument('--clg-path', type=str, default='./models/poly_clf_time/01_64sde_cls_with_time2__vtype_VAE_b_0.1_5.0_',
    #                     help='classifier guidance path [default: "./models/poly_clf_time/01_64sde_cls_with_time2__vtype_VAE_b_0.1_5.0_"]')
    # parser.add_argument('--clg-path', type=str, default='./models/poly_clf_time/0123456789_64sde__vtype_VAE_b_0.1_5.0_',
    #                     help='classifier guidance path [default: "./models/poly_clf_time/0123456789_64sde__vtype_VAE_b_0.1_5.0_"]')
    # parser.add_argument('--clg-path', type=str, default='./models/poly_clf_time/0123456789_64sde_cls_with_time2__vtype_VAE_b_0.1_5.0_',
    #                     help='classifier guidance path [default: "./models/poly_clf_time/0123456789_64sde_cls_with_time2__vtype_VAE_b_0.1_5.0_"]')
    parser.add_argument('--clg-path', type=str, default='./models/poly_clf_time/0123456789_64sde_cls_with_time2_RandN__vtype_VAE_b_0.1_5.0_',
                        help='classifier guidance path [default: "./models/poly_clf_time/0123456789_64sde_cls_with_time2_RandN__vtype_VAE_b_0.1_5.0_"]')
    parser.add_argument('--cl-s', type=float, default=1.0,
                        help='classifier guidance scale  [default: 1.0]')
    
    parser.add_argument('--eval-only', type=int, default=0, 
                        help='if 1, no training, eval only')
    # parser.add_argument('--score-path', type=str, default='./models/sde_poly/01_64_VAE_sde_unet__vtype_VAE_dim_32_N_100_b_0.1_5.0__ImpSamp__n_obs_',
    #                     help='score path [default: "./models/sde_poly/01_64_VAE_sde_unet__vtype_VAE_dim_32_N_100_b_0.1_5.0__ImpSamp__n_obs_"]')
    parser.add_argument('--score-path', type=str, default='./models/sde_poly/0123456789_64_VAE_sde_unet__vtype_VAE_dim_128_N_100_b_0.1_5.0__pc_True_snr_0.16__last_epoch',
                        help='score path [default: "./models/sde_poly/0123456789_64_VAE_sde_unet__vtype_VAE_dim_128_N_100_b_0.1_5.0__pc_True_snr_0.16__last_epoch"]')
    # parser.add_argument('--score-path', type=str, default='./models/sde_poly/0123456789_64_VAE_sde_unet__vtype_VAE_dim_64_N_100_b_0.1_5.0__pc_True_snr_0.16_',
    #                     help='score path [default: "./models/sde_poly/0123456789_64_VAE_sde_unet__vtype_VAE_dim_64_N_100_b_0.1_5.0__pc_True_snr_0.16_"]')
    
    args = parser.parse_args()

    if args.vae_type == "VAE":
        vae_paths = [args.p0_path, args.p1_path, args.p2_path, args.p3_path, args.p4_path, args.p5_path, args.p6_path, args.p7_path, args.p8_path, args.p9_path]
    elif args.vae_type == "AE":
        vae_paths = [args.p0_path_ae, args.p1_path_ae, args.p2_path_ae, args.p3_path_ae, args.p4_path_ae, args.p5_path_ae, args.p6_path_ae, args.p7_path_ae, args.p8_path_ae, args.p9_path_ae]
    else:
        raise Exception("Wrong VAE type")
    
    run(args.epochs, args.batch_size, args.lr, args.size_z, args.allmods, args.savefolder, vae_paths, args.sm_type, args.unq_name, args.cuda, args.pclf_path, \
        args.vae_type, args.beta0, args.beta1, args.N, args.T, args.ll_weighting, args.noise_obs, args.pc, args.n_steps, args.target_snr, args.im_sample, \
        args.use_clg, args.clg_path, args.cl_s, args.eval_only, args.score_path)


