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
from numpy import prod
import math
from mmplus_model_orig import *


from polymnist_dataset import get_train_test_dataset_upd10_28x28
from mopoe_model import MOPOEPolyRes, MMVAEPolyRes, MVPolyRes, MVTPolyRes
from pytorch_fid.fid_score import calculate_fid_given_paths
from polymnist_model import PMCLF

import shutil
from utils import *

def get_train_test_dataloader_upd10_28x28(batch_size):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_28x28()
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

# def get_train_test_dataloader_upd10_28x28(batch_size):
#     paired_val_dataset = test_dataset_upd10_32x32(test=False)
#     # train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     # test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     return val_dataloader, val_dataloader, val_dataloader

# def calc_kl_loss(mu, logvar, cons=1):
#     return cons * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[0]

# def calc_kl_loss_2(mu0, logvar0, mu1, logvar1, cons=1):
#     kl2 = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
#     return cons * kl2 / mu0.shape[0]

# def image_loss(x_hat, x, cons=1):
#     mse = nn.MSELoss(reduction='sum')
#     recon_loss = mse(x_hat, x) / x.shape[0]
#     return cons*recon_loss

# def total_rec_loss(outs, inputs):
#     rec_loss = 0
#     for i in range(len(outs)):
#         rec_loss += image_loss(outs[i], inputs[i])
#     # return (1/len(outs)) * rec_loss
#     return 1 * rec_loss

# def total_kl_loss(mus, logvars, cons=1):
#     kl_losses = 0
#     for i in range(len(mus)):
#         kl_losses += calc_kl_loss(mus[i], logvars[i])
#     return (1/len(mus)) * cons * kl_losses

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
    # print("shapes: ", qu_xs.shape, px_us.shape, uss.shape)
    # raise Exception("shape ")
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


def train_model(train_loader, model, mix_type, optimizer, all_mods, K, estimator, device, kl_cons):
    losses = 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):

        inputs = [images['m' + mod].to(device) for mod in all_mods]

        if estimator == 'iwae':
            total_loss = -m_iwae(model, inputs, K, test=False)
        elif estimator == 'dreg':
            total_loss = -m_dreg(model, inputs, K, test=False)
        losses += total_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    end_time = time.time()
    losses /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return losses

def evaluate_model(val_loader, model, mix_type, all_mods, K, estimator, device, epoch, vae_type, unq_name, show=True, kl_cons=1):
    with torch.no_grad():
        losses = 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, target) in enumerate(val_loader):

            inputs = [images['m' + mod].to(device) for mod in all_mods]

            if estimator == 'iwae':
                total_loss = -m_iwae(model, inputs, K, test=True)
            elif estimator == 'dreg':
                total_loss = -m_dreg(model, inputs, K, test=True)
            losses += total_loss.item()

            if batch_idx == 0:
                random_idx = torch.randint(0, len(all_mods), (1,)).item()
                a = model.cond_gen([random_idx], [inp[:10] for inp in inputs])
                concat_tensor = torch.cat(a, dim=-4).squeeze()
                
                concat_tensor = torchvision.utils.make_grid(concat_tensor, nrow=10)
                plt.imshow(concat_tensor.permute(1,2,0).detach().cpu().numpy())
                plt.title('  Given ' + str(random_idx))
                plt.axis("off")
                plt.savefig('./images/' + mix_type + '/' + 'cond_'  + mix_type + all_mods + "_vae_" + vae_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx)  + unq_name + ".png")
            
        end_time = time.time()
        losses /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        # print("fids: ", fids, flush=True)
        print("Validation loss: ", losses, flush=True)
        return losses  

def calc_poly_cond(test_loader, sample_path, mix_vae, mix_type, predicted_mods, all_mods, p_clf, size_z, device, write_input=True):
    with torch.no_grad():
        mix_vae.eval()
        cond_accuracies = {}
        for pred in predicted_mods:
            cond_accuracies[pred] = 0
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in all_mods:
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            present_mod = ''.join([i for i in all_mods if i not in predicted_mods])
            present_mod_list = [int(m) - int(all_mods[0]) for m in present_mod]
            all_predicted = mix_vae.cond_gen(present_mod_list, [p[mod] for mod in all_mods])

            for mod in predicted_mods:
                p_out[mod] = all_predicted[int(mod) - int(all_mods[0])]
                predicted_out = p_clf(p_out[mod].view(-1,3,28,28))
                predicted_out = torch.argmax(predicted_out, 1)
                
                cond_acc = torch.sum(predicted_out == target).item()
                cond_acc = cond_acc / p[mod].shape[0]
                cond_accuracies[mod] += cond_acc
                
                if write_input:
                    save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + mod + '_' + ''.join([i for i in all_mods if i not in predicted_mods])] + str(batch_idx) + '_')

        for mod in cond_accuracies.keys():
            cond_accuracies[mod] /= len(test_loader)

        fid_scores_cond = {}
        for pred in predicted_mods:
            fid_p = calculate_fid_given_paths([sample_path['p'+pred], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mods])]], 256, device, 2048, 2)
            fid_scores_cond[pred] = fid_p

        print("Cond Coherence: ", cond_accuracies, flush=True)
        return fid_scores_cond


def run(epochs, batch_size, lr, size_z, beta, unq_name, vae_type, mix_type, all_mods, cuda_num, estimator, k, pclf_path):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    kl_cons = beta
    unq_name += '_k_' + str(k) + estimator 
    print('vars: ', epochs, batch_size, lr, size_z, kl_cons, unq_name, vae_type, mix_type, estimator, k)
    train_losses, val_losses = [], []
    
    for p in ['./models/' + mix_type + '/', './plots/' + mix_type + '/', './images/' + mix_type + '/']:
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, cuda_num, flush=True)
    device = torch.device("cuda:"+str(cuda_num))
    print("device: ", torch.cuda.get_device_properties(device), flush=True)
    b_size = batch_size

    class Params():
        latent_dim_w = size_z // 2
        latent_dim_z = size_z // 2
        model = 'polymnist_10modalities'
        obj = estimator
        K = k
        batch_size = b_size
        epochs = 200
        beta = kl_cons
        learn_prior_w_polymnist = True
        variant = 'mmvaeplus'
        tmpdir = '/tmp/'
        no_cuda = False
        n_mod = len(all_mods)

    params = Params()
    mix_vae = PolyMNIST_10modalities(params)
    
    # optimizer = torch.optim.Adam(mix_vae.parameters(), lr=lr, amsgrad=True)
    optimizer = torch.optim.Adam(mix_vae.parameters(), lr=lr)
    mix_vae = mix_vae.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print("Mods: ", all_mods, flush=True)
    print("K: ", k, "estimator: ", estimator)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path, map_location=device))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()
    
    train_dataloader, val_dataloader, _ = get_train_test_dataloader_upd10_28x28(batch_size)
    print("data loaded")

    best_average_fid = 1000
    best_epoch = 0

    sample_path = {}
    temp_rand = torch.randint(1000000, size=(1,)).item()
    temp_dir_name = './t_' + mix_type + str(kl_cons) + all_mods + '_' + str(temp_rand) + '/'
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

    for p in sample_input_path + sample_output_path:
        if not os.path.exists(p):
            os.makedirs(p)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, mix_vae, mix_type, optimizer, all_mods, k, estimator, device, kl_cons)
        validation_loss = evaluate_model(val_dataloader, mix_vae, mix_type, all_mods, k, estimator, device, epoch, vae_type, unq_name, True, kl_cons)
        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        
        if (epoch + 1) % 50 == 0:
            # calculate average cond FID
            average_fid = 0 
            for mod in all_mods:
                fid_score = calc_poly_cond(val_dataloader, sample_path, mix_vae, mix_type, mod, all_mods, poly_clf, size_z, device, write_input=True)
                print("fid ", mod, fid_score[mod], flush=True)
                average_fid += fid_score[mod]
            
            average_fid /= len(all_mods)
            print("Epoch: ", epoch+1, " Average cond-FID: ", average_fid, flush=True)

            if average_fid < best_average_fid:
                best_average_fid = average_fid
                best_epoch = epoch + 1

            torch.save({
            'epoch': epoch,
            'model_state_dict': mix_vae.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'beta': kl_cons,
            'size_z': size_z,
            }, "./models/" + mix_type + '/' + mix_type + "NEW" + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z) + all_mods + '_' + unq_name + 'fid_' + str(average_fid)[:4] + '_epoch_' + str(epoch + 1))
            print('Model saved', flush=True)
            print(" ")

    print("best average fid: ", best_average_fid, ' at epoch: ', best_epoch, flush=True)

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot_train_val(train_losses, val_losses, 'Total loss', ['Train', 'Val'], \
        './plots/' + mix_type + '/' + mix_type + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z) + all_mods + '_' + unq_name)  
    
    shutil.rmtree(temp_dir_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--all-mods', type=str, default='0123456789',
                        help='all mods [default: "0123456789"]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--cuda', type=int, default=3,
                        help='number of cuda_gpu [default: 3]')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta value for kl loss [default: 0.5]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--unq-name', type=str, default='',
                        help='identifier name for saving [default: ""]')
    parser.add_argument('--estimator', type=str, default='iwae',
                        help='estimator type iwae or dreg [default: "iwae"]')
    parser.add_argument('--k', type=int, default=1,
                        help='number of k in the estimator [default: 1]')
    parser.add_argument('--mix-type', type=str, default='mmplusOrig',
                        help='mixture type [default: "mmplusOrig"]')
    parser.add_argument('--vae-type', type=str, default='res',
                        help='vae type [default: "res"]')
    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')
 
    
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.unq_name, args.vae_type, args.mix_type, args.all_mods, args.cuda, args.estimator, args.k, args.pclf_path)