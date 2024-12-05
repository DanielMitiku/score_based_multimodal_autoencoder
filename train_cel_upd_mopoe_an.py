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
from numpy import prod
import math

from mopoe_model import MOPOECeleb, MVTCeleb
from mmplus_model_cel import MMPLUSCeleba
from celeba_hq_mask_dataset import CelebAHQMaskDS
from utils import *

from sklearn.metrics import f1_score
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil

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

def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader

# def get_train_test_dataloader(batch_size, size):
#     # train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
#     val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

#     # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return val_dataloader, val_dataloader

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def _m_iwae(model, x, K=1, test=False):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    if test:
        qu_xs, px_us, uss = model.reconstruct_and_cross_reconstruct_forw(x)
    else:
        qu_xs, px_us, uss = model(x, K)
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qu_x in enumerate(qu_xs):
        lpu = model.pu(*model.pu_params).log_prob(uss[r]).sum(-1)
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]
        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta*(lpu - lqz_x - lqw_x)
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size

def m_iwae(model, x, K=1, test=False):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K, test=test) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()

def _m_dreg(model, x, K=1, test=False):
    """DReG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    if test:
        qu_xs, px_us, uss = model.reconstruct_and_cross_reconstruct_forw(x)
    else:
        qu_xs, px_us, uss = model(x, K)
    qu_xs_ = [vae.qu_x(*[p.detach() for p in vae.qu_x_params]) for vae in model.vaes]
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs_):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qu_x in enumerate(qu_xs_):
        lpu = model.pu(*model.pu_params).log_prob(uss[r]).sum(-1)
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        # for d, px_U in enumerate(px_us[r]):
        #     print('xd: ', x[d].shape, flush=True)
        #     print('pxu params: ', px_U.mean.shape, flush=True)
        #     print("pxu shape: ", px_U.log_prob(x[d]).shape, flush=True)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]
        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta*(lpu - lqz_x - lqw_x)
        lws.append(lw)
    return torch.stack(lws), torch.stack(uss)

def m_dreg(model, x, K=1, test=False):
    """Computes DReG estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, uss = zip(*[_m_dreg(model, _x, K, test=test) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    uss = torch.cat(uss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()

def calc_kl_loss(mu, logvar, cons=1):
    return cons * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[0]

def calc_kl_loss_2(mu0, logvar0, mu1, logvar1, cons=1):
    kl2 = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
    return cons * kl2 / mu0.shape[0]

def attr_loss(x_hat, x):
    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    recon_loss = bce_logit_loss(x_hat, x)
    # mse = nn.MSELoss(reduction='sum')
    return recon_loss/ x_hat.shape[0]

def image_loss(x_hat, x, cons=1):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x) / x.shape[0]
    return cons*recon_loss

def total_rec_loss(outs, inputs):
    rec_loss = 0
    for i in range(len(outs)):
        if i == len(outs) - 1:
            rec_loss += attr_loss(outs[i], inputs[i])
        else:    
            rec_loss += image_loss(outs[i], inputs[i])
    # return (1/len(outs)) * rec_loss
    return 1 * rec_loss

def total_kl_loss(mus, logvars, cons=1):
    kl_losses = 0
    for i in range(len(mus)):
        kl_losses += calc_kl_loss(mus[i], logvars[i])
    return (1/len(mus)) * cons * kl_losses

def mvae_loss(elbo_terms, inputs, kl_cons=1):
    assert len(elbo_terms) == (len(inputs) + 2)
    rec, kl = 0, 0
    # calc individual elbo loss
    for i in range(len(inputs)):
        elbo = elbo_terms[i]
        kl += calc_kl_loss(elbo[0], elbo[1])
        rec += image_loss(elbo[2], inputs[i])

    # calc joint elbo loss
    kl += calc_kl_loss(elbo_terms[len(inputs)][0], elbo_terms[len(inputs)][1])
    for i in range(len(inputs)):
        if i == len(inputs) - 1:
            rec += attr_loss(elbo_terms[len(inputs)][2][i], inputs[i])
        else:    
            rec += image_loss(elbo_terms[len(inputs)][2][i], inputs[i])

    # calc kth elbo loss
    kl += calc_kl_loss(elbo_terms[len(inputs)+1][0], elbo_terms[len(inputs)+1][1])
    k_idxs = elbo_terms[len(inputs)+1][2]
    k_outs = elbo_terms[len(inputs)+1][3]
    for i, k_idx in enumerate(k_idxs):
        rec += image_loss(k_outs[i], inputs[k_idx])

    return rec, kl_cons * kl

def mvt_loss(elbo_terms, inputs, kl_cons=1, alpha=0.9):
    assert len(elbo_terms) == (len(inputs) + 1)

    # calc joint elbo loss
    kl_joint = calc_kl_loss(elbo_terms[len(inputs)][0], elbo_terms[len(inputs)][1])
    rec = 0
    for i in range(len(inputs)):
        if i == len(inputs) - 1:
            rec += attr_loss(elbo_terms[len(inputs)][2][i], inputs[i])
        else:    
            rec += image_loss(elbo_terms[len(inputs)][2][i], inputs[i])

    kl2 = 0
    # calc individual elbo loss
    for i in range(len(inputs)):
        elbo = elbo_terms[i]
        kl2 += calc_kl_loss_2(elbo_terms[len(inputs)][0], elbo_terms[len(inputs)][1], elbo[0], elbo[1])

    # total_loss
    rec_weight = (len(inputs) - alpha) / len(inputs)
    cvib_weight = alpha / len(inputs) 
    vib_weight = 1 - alpha 
    
    kld_weighted = cvib_weight * kl2 + vib_weight * kl_joint
    return rec_weight * rec, kl_cons * kld_weighted


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
        given_str = ''

        for mod in given:
            if mod == '0':
                given_str += 'Image '
            elif mod == '1':
                given_str += 'Mask '
            else:
                given_str += 'Attr '
        print("GIVEN: ", given_str)

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


def train_model(train_loader, model, attr_visible, mix_type, optimizer, K, estimator, device, kl_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = images.to(device)
        masks = masks.to(device)
        target = target.to(device)[:, attr_visible]
        inputs = [input, masks, target.float()]

        if mix_type == 'mmplus':
            if estimator == 'iwae':
                total_loss = -m_iwae(model, inputs, K, test=False)
            elif estimator == 'dreg':
                total_loss = -m_dreg(model, inputs, K, test=False)
            losses['total'] += total_loss.item()
        else:
            if mix_type == 'mopoe' or mix_type == 'mmvae':
                outs, mus, logvars = model(inputs)
                rec, kl = total_rec_loss(outs, inputs), total_kl_loss(mus, logvars, kl_cons)
            elif mix_type == 'mvae':
                elbo_terms = model(inputs)
                rec, kl = mvae_loss(elbo_terms, inputs, kl_cons)
            elif mix_type == 'mvt':
                elbo_terms = model(inputs)
                rec, kl = mvt_loss(elbo_terms, inputs, kl_cons)

            losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()
            total_loss = rec + kl
            losses['total'] += total_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    end_time = time.time()
    for loss in losses:
        losses[loss] /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return list(losses.values())

def evaluate_model(val_loader, model, attr_visible, mix_type, epoch, K, estimator, device, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, masks, target) in enumerate(val_loader):

            input = images.to(device)
            masks = masks.to(device)
            target = target.to(device)[:, attr_visible]
            inputs = [input, masks, target.float()]

            if mix_type == 'mmplus':
                if estimator == 'iwae':
                    total_loss = -m_iwae(model, inputs, K, test=True)
                elif estimator == 'dreg':
                    total_loss = -m_dreg(model, inputs, K, test=True)
                losses['total'] += total_loss.item()
            else:
                if mix_type == 'mopoe' or mix_type == 'mmvae':
                    outs, mus, logvars = model(inputs)
                    rec, kl = total_rec_loss(outs, inputs), total_kl_loss(mus, logvars, kl_cons)
                elif mix_type == 'mvae':
                    elbo_terms = model(inputs)
                    rec, kl = mvae_loss(elbo_terms, inputs, kl_cons)
                elif mix_type == 'mvt':
                    elbo_terms = model(inputs)
                    rec, kl = mvt_loss(elbo_terms, inputs, kl_cons)

                losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()
                total_loss = rec + kl
                losses['total'] += total_loss.item()

        # Plot random input and output
        sample_idx = torch.randint(inputs[0].shape[0], size=(1,)).item()
        sample_out = model.cond_gen([2], [input, masks, target.float()])
        # if mix_type == 'mvae':
        #     sample_out = torch.cat([elbo_terms[len(inputs)][2][i][sample_idx].permute(1,2,0) for i in range(len(inputs))], dim=0)
        
        if mix_type == 'mmplus':
            sigmoid_outputs = sample_out[2].detach().cpu()
        else:
            sigmoid_outputs = torch.sigmoid(sample_out[2]).detach().cpu()
        predicted_att = np.round(sigmoid_outputs)

        tar_str, pred_str = 'T: ', 'P: '
        for ind,att in enumerate(target[0]):
            if int(att) == 1:
                tar_str += new_id_to_attr[ind] + ' '
        for ind,att in enumerate(predicted_att[0]):
            if int(att) == 1:
                pred_str += new_id_to_attr[ind] + ' '

        plt.figure()
        grid = torchvision.utils.make_grid(torch.cat([input[0].unsqueeze(0), sample_out[0][0].unsqueeze(0)],dim=0), nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig('./images/chq_' + mix_type + '/' + 'img_' + mix_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx) + ".png")
        plt.figure()
        grid = torchvision.utils.make_grid(torch.cat([masks[0].unsqueeze(0), sample_out[1][0].unsqueeze(0)],dim=0), nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig('./images/chq_' + mix_type + '/' + 'mask_' + mix_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx) + ".png")
        plt.figure()
        plt.text(0.05,0.5,tar_str + '\n' + pred_str, fontsize='xx-small', fontfamily='monospace')
        plt.savefig('./images/chq_' + mix_type + '/' + 'attr' + mix_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx) + ".png")
        plt.close('all') 


        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        # print("fids: ", fids, flush=True)
        print("Validation loss: ", losses, flush=True)
        return list(losses.values())   


def run(epochs, batch_size, lr, size_z, beta, unq_name, vae_type, mix_type, cuda_num, k, estimator):
    kl_cons = beta
    attr_visible = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
    print('vars: ', epochs, batch_size, lr, size_z, kl_cons, unq_name, vae_type, mix_type, attr_visible)
    train_losses, val_losses = [], []

    att_threshold = 0.5

    for p in ['./models/chq_' + mix_type + '/', './plots/chq_' + mix_type + '/', './images/chq_' + mix_type + '/']:
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, cuda_num, flush=True)
    device = torch.device("cuda:"+str(cuda_num))
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    rand_num = str(int(torch.rand(1)*10000))
    temp_dir_name = './samp/t_' + mix_type + '_' + str(rand_num) + '/'
    print('temp dir: ', temp_dir_name, flush=True)

    sample_path = {'in_image': temp_dir_name + 'temp_hq_in' + rand_num + '/', 
            'out_image_' + mix_type: temp_dir_name + 'temp_hq_out_' + mix_type + rand_num + '/', }
    
    for p in sample_path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    if vae_type == 'res' and mix_type == 'mopoe':
        #sm
        enc_channel_list1 = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
        dec_channel_list1 = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
        enc_channel_list2 = [(64,128,128,4), (128,256,256,4)]
        dec_channel_list2 = [(256,256,128,4), (128,128,64,4)]
        size_in = 128
        img_ch = 3
        mix_vae = MOPOECeleb(enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=size_z, size_in=size_in, img_ch=img_ch, mask_ch=1)
    elif mix_type == 'mvt':
        enc_channel_list1 = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
        dec_channel_list1 = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
        enc_channel_list2 = [(64,128,128,4), (128,256,256,4)]
        dec_channel_list2 = [(256,256,128,4), (128,128,64,4)]
        size_in = 128
        img_ch = 3
        mix_vae = MVTCeleb(enc_channel_list1, dec_channel_list1, enc_channel_list2, dec_channel_list2, size_z=size_z, size_in=size_in, img_ch=img_ch, mask_ch=1)
    elif mix_type == 'mmplus':
        print('K is ', k, ' Estimator: ', estimator)
        b_size = batch_size
        class Params():
            latent_dim_w = size_z // 2
            latent_dim_z = size_z // 2
            model = 'celebhq'
            obj = estimator
            K = k
            batch_size = b_size
            epochs = 300
            beta = kl_cons
            learn_prior_w_polymnist = True
            variant = 'mmvaeplus'
            tmpdir = '/tmp/'
            no_cuda = False
            n_mod = 3
        params = Params()
        mix_vae = MMPLUSCeleba(params)
    else:
        raise Exception('Not Implemented')
    
    optimizer = torch.optim.Adam(mix_vae.parameters(), lr=lr)
    mix_vae = mix_vae.to(device)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, 128)
    
    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, mix_vae, attr_visible, mix_type, optimizer, k, estimator, device, kl_cons)
        validation_loss = evaluate_model(val_dataloader, mix_vae, attr_visible, mix_type, epoch, k, estimator, device, kl_cons)
        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        # scheduler.step(validation_loss[0])

        if (epoch + 1) % 100 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': mix_vae.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss[0],
            'val_loss': validation_loss[0],
            'beta': kl_cons,
            'size_z': size_z,
            }, "./models/chq_" + mix_type + "/" + "celebhqAN_" + mix_type + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z) + '_epoch_' + str(epoch + 1))
            print('Model saved', flush=True)

            for given in ['', '0', '1', '2', '01', '02', '12']:
                evaluate_mopoe(val_dataloader, mix_vae, mix_type, attr_visible, att_threshold, size_z, device, sample_path, given)


    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/chq_' + mix_type + '/' + 'celebhqAN_' + mix_type + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z))  
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training [default: 128]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta value for kl loss [default: 1]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    parser.add_argument('--unq-name', type=str, default='',
                        help='identifier name for saving [default: ""]')
    parser.add_argument('--mix-type', type=str, default='mopoe',
                        help='mixture type [default: "mopoe"]')
    parser.add_argument('--vae-type', type=str, default='res',
                        help='vae type [default: "res"]')
    parser.add_argument('--cuda', type=int, default=3,
                        help='number of cuda_gpu [default: 3]')
    parser.add_argument('--estimator', type=str, default='iwae',
                        help='estimator type iwae or dreg [default: "iwae"]')
    parser.add_argument('--k', type=int, default=1,
                        help='number of k in the estimator [default: 1]')
    
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.unq_name, args.vae_type, args.mix_type, args.cuda, args.k, args.estimator)


