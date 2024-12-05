import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np
import os

from polymnist_dataset import get_train_test_dataset_upd10_32x32, test_dataset_upd10_32x32
from h_vae_model_copy import ResVAE
from lat_sm2_model import LSMPoly64_sm, Poly_sm, Poly_sm2
from polymnist_model import PMCLF

from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_32x32()
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

def get_train_test_dataloader_upd10_32x32_val(batch_size):
    paired_val_dataset = test_dataset_upd10_32x32(test=False)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return val_dataloader

def sm_loss(score_net, input):
    grad_energy = score_net(input)
    loss1 = torch.square(torch.norm(grad_energy, dim=-1))

    # loss2 to calculate the trace of the gradients with respect to each input
    loss2 = torch.zeros(input.shape[0], device=input.device)
    for i in range(input.shape[1]):
        i_grad = torch.autograd.grad(grad_energy[:,i].sum(), input, create_graph=True, retain_graph=True)[0].view(input.shape[0],-1)[:,i]
        loss2 += i_grad
    
    return (0.5 * (loss1 + loss2)).mean()

def dsm_loss(sm_model, q_z, sigma=1):
    noise = sigma * torch.normal(mean=0, std=1, size=q_z.shape, device=q_z.device)
    perturbed_q_z = q_z + noise
    sm_out = sm_model(perturbed_q_z)
    if isinstance(sigma, int): 
        loss = 0.5 * (sigma ** 2) * ((sm_out + (noise/sigma**2)) ** 2).sum(dim=-1)
    else:
        loss = 0.5 * (sigma.squeeze() ** 2) * ((sm_out + (noise/sigma**2)) ** 2).sum(dim=-1)
    return loss.mean()

def ssm_loss(score_net, input):
    grad_energy = score_net(input)
    proj_vec = torch.randn_like(input)
    loss1 = 0.5 * torch.square(torch.sum(grad_energy * proj_vec, dim=-1))

    grad_score_proj = torch.autograd.grad(torch.sum(grad_energy * proj_vec), input, create_graph=True)[0]
    loss2 = torch.sum(grad_score_proj * proj_vec, dim=-1)
    
    return (loss1 + loss2).mean()

def recon_loss(x, x_hat):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    return recon_loss / x.shape[0]

def train_model(train_loader, pvae_dict, pvae_opt, sm_model, drop_p, all_mods, n_comp, lr1, lr2, device, size_z=64):
    losses = {}
    len_losses = {}
    for mod in sorted(pvae_dict.keys()):
        pvae_dict[mod].train()
        losses[mod] = 0
        len_losses[mod] = 0
    sm_model.eval()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):
        with torch.no_grad():
            trained_mods = ''
            p = {}
            z = {}
            # mod_to_keep = str(batch_idx % 10)
            for mod in pvae_dict.keys():
                p[mod] = images['m'+mod].to(device)
                # if mod == mod_to_keep:
                if (np.random.uniform() > drop_p):
                    z[mod] = pvae_dict[mod].reparametrize(*pvae_dict[mod].encoder(p[mod]))
                else:
                    trained_mods += mod
                    z[mod] = torch.normal(mean=0, std=1, size=(p[mod].shape[0],size_z), device=device)

            # # stack zs
            # z = torch.cat([z[mod] for mod in sorted(pvae_dict.keys())], dim=1).detach()

            for i in range(n_comp):
                z_in = torch.cat([z[mod] for mod in all_mods], dim=1)
                sm_out = sm_model(z_in)

                for ind,mod in enumerate(sorted(pvae_dict.keys())):
                    if mod in trained_mods:
                        z[mod] = z[mod] + (lr1 * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
        
        with torch.enable_grad():
            for mod in trained_mods:
                out = pvae_dict[mod].decoder(z[mod])
                loss = recon_loss(out, p[mod])
                losses[mod] += loss.item()
                len_losses[mod] += 1

                pvae_opt[mod].zero_grad()
                loss.backward()
                pvae_opt[mod].step()

    end_time = time.time()
    for mod in all_mods:
        if len_losses[mod] > 0:
            losses[mod] /= len_losses[mod]
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return losses

def evaluate(val_loader, pvae_dict, sm_model, drop_p, all_mods, n_comp, lr1, lr2, device, size_z=64):
    losses = {}
    len_losses = {}
    for mod in sorted(pvae_dict.keys()):
        pvae_dict[mod].train()
        losses[mod] = 0
        len_losses[mod] = 0
    sm_model.eval()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(val_loader):
        with torch.no_grad():
            trained_mods = ''
            p = {}
            z = {}
            # mod_to_keep = str(batch_idx % 10)
            for mod in pvae_dict.keys():
                p[mod] = images['m'+mod].to(device)
                # if mod == mod_to_keep:
                if (np.random.uniform() > drop_p):
                    z[mod] = pvae_dict[mod].reparametrize(*pvae_dict[mod].encoder(p[mod]))
                else:
                    trained_mods += mod
                    z[mod] = torch.normal(mean=0, std=1, size=(p[mod].shape[0],size_z), device=device)

                # # stack zs
                # z = torch.cat([z[mod] for mod in sorted(pvae_dict.keys())], dim=1).detach()

            for i in range(n_comp):
                z_in = torch.cat([z[mod] for mod in all_mods], dim=1)
                sm_out = sm_model(z_in)

                for ind,mod in enumerate(sorted(pvae_dict.keys())):
                    if mod in trained_mods:
                        z[mod] = z[mod] + (lr1 * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
        
        with torch.no_grad():
            for mod in trained_mods:
                out = pvae_dict[mod].decoder(z[mod])
                loss = recon_loss(out, p[mod])
                losses[mod] += loss.item()
                len_losses[mod] += 1

    end_time = time.time()
    for mod in all_mods:
        if len_losses[mod] > 0:
            losses[mod] /= len_losses[mod]
    print("Validation TIME TAKEN: ", end_time - start_time, flush=True)
    print("Validation loss: ", losses, flush=True)
    return losses


def calc_poly_cond(test_loader, vae_dict, sm_model, drop_p, all_mods, p_clf, lr1, lr2, n_comp, size_z, device, use_mean=True, schedule=False):
    with torch.no_grad():
        for vae in vae_dict.values():
            vae.eval()
        cond_accuracies = {}
        len_losses = {}

        for pred in all_mods:
            len_losses[pred] = 0
            cond_accuracies[pred] = 0
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in sorted(vae_dict.keys()):
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            z = {}

            predicted_mods = ''
            for mod in all_mods:
                if (np.random.uniform() < drop_p):
                    predicted_mods += mod
                    len_losses[mod] += 1

            for key in sorted(vae_dict.keys()):
                if key in predicted_mods:
                    z[key] = torch.normal(mean=0, std=1, size=(p[key].shape[0],size_z), requires_grad=True, device=device)
                if key not in predicted_mods:
                    if use_mean:
                        z[key] = vae_dict[key].encoder(p[key])[0]
                    else:
                        z[key] = vae_dict[key].reparametrize(*vae_dict[key].encoder(p[key])) 

            for i in range(n_comp):
                z_in = torch.cat([z[mod] for mod in all_mods], dim=1)
                sm_out = sm_model(z_in)

                for ind,mod in enumerate(sorted(vae_dict.keys())):
                    if mod in predicted_mods:
                        if not schedule:
                            z[mod] = z[mod] + (lr1 * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
                        else:
                            z[mod] = z[mod] + (lr1 * ((i+1)/n_comp) * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
                            if i == n_comp - 1:
                                z[mod] = z[mod] + (1 * ((i+1)/n_comp) * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)


            for mod in predicted_mods:
                p_out[mod] = vae_dict[mod].decoder(z[mod])
                predicted_out = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
                predicted_out = torch.argmax(predicted_out, 1)
                
                cond_acc = torch.sum(predicted_out == target).item()
                cond_acc = cond_acc / p[mod].shape[0]
                cond_accuracies[mod] += cond_acc
                
        for mod in cond_accuracies.keys():
            if len_losses[mod] > 0:
                cond_accuracies[mod] /= len_losses[mod]
        print("Cond Coherence: ", cond_accuracies, flush=True)
        return cond_accuracies   


def run(epochs, batch_size, lr, size_z, all_mods, savefolder, model_paths, sm_type, sm_path, pclf_path, unq_name, drop_p):
    unq_name += '_' + str(drop_p) + '_'
    print('vars: ', epochs, batch_size, lr, size_z, sm_type, unq_name, all_mods, savefolder, drop_p, flush=True)
    train_losses, val_losses = {}, {}
    lr1 = 0.2
    lr2 = 0.02
    n_comp = 40

    savefolder += '/'
    save_paths = {'model': './models/' + savefolder, 'plot': './plots/' + savefolder, 'image': './images/' + savefolder}
    for p in save_paths.values():
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:5")
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()
    
    enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
    dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
    size_in = 32
    img_ch = 3
    pvae_dict = {}
    pvae_opt = {}
    n_mod = len(all_mods)
    
    for ind, model_path in enumerate(model_paths):
        if str(ind) in all_mods:
            pmvae = ResVAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
            pmvae.load_state_dict(torch.load(model_path)['model_state_dict'])
            pmvae = pmvae.to(device)
            pvae_dict[str(ind)] = pmvae

    for mod in all_mods:
        pvae_opt[mod] = torch.optim.Adam(pvae_dict[mod].parameters(), lr=lr)
        train_losses[mod] = 1e10
        val_losses[mod] = 1e10

    score_model = LSMPoly64_sm(n_mod, size_z)
    score_model.load_state_dict(torch.load(sm_path)['model_state_dict'])
    score_model = score_model.to(device)
    
    train_dataloader, val_dataloader, _ = get_train_test_dataloader_upd10_32x32(batch_size)
    val_b1_dataloader = get_train_test_dataloader_upd10_32x32_val(32)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        calc_poly_cond(val_b1_dataloader, pvae_dict, score_model, drop_p, all_mods, poly_clf, lr1, lr2, n_comp, size_z, device)
        print(' ')

        training_loss = train_model(train_dataloader, pvae_dict, pvae_opt, score_model, drop_p, all_mods, n_comp, lr1, lr2, device, size_z)
        validation_loss = evaluate(val_dataloader, pvae_dict, score_model, drop_p, all_mods, n_comp, lr1, lr2, device, size_z)
        # print(' ', flush=True)


        for mod in all_mods:
            if training_loss[mod] < train_losses[mod]:
                train_losses[mod] = training_loss[mod]
            if validation_loss[mod] < val_losses[mod]:
                val_losses[mod] = validation_loss[mod]
                torch.save({
                'model_state_dict': pvae_dict[mod].state_dict(),
                'train_loss': training_loss[mod],
                'val_loss': validation_loss[mod],
                'size_z': size_z,
                }, save_paths['model'] + mod + '_' + str(size_z) + str(unq_name))
                print('Model saved, mod ', mod, flush=True)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train [default: 50]')
    parser.add_argument('--upd', type=str, default='',
                        help='updated polymnist dataset [default: ]')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate [default: 0.0001]')
    parser.add_argument('--sm-type', type=str, default='dsm',
                        help='loss type [default: "dsm"]')
    parser.add_argument('--drop-p', type=float, default=0.5,
                        help='drop prob [default: 0.5]')
    parser.add_argument('--allmods', type=str, default='0123456789',
                        help='Mods to train on [default: "0123456789"]')
    parser.add_argument('--savefolder', type=str, default='finetune_vae',
                        help='folder name to save output [default: "finetune_vae"]')
    parser.add_argument('--unq-name', type=str, default='poly_finetuned_res_',
                        help='identifier name for saving [default: "poly_finetuned_res_"]')
    parser.add_argument('--p0-path', type=str, default='./models/polyupd10_m0/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m0/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p1-path', type=str, default='./models/polyupd10_m1/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m1/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p2-path', type=str, default='./models/polyupd10_m2/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m2/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p3-path', type=str, default='./models/polyupd10_m3/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m3/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p4-path', type=str, default='./models/polyupd10_m4/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m4/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p5-path', type=str, default='./models/polyupd10_m5/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m5/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p6-path', type=str, default='./models/polyupd10_m6/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m6/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p7-path', type=str, default='./models/polyupd10_m7/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m7/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p8-path', type=str, default='./models/polyupd10_m8/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m8/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--p9-path', type=str, default='./models/polyupd10_m9/polyupd10_res_beta_0.5__64',
                        help='multimodal model path [default: "./models/polyupd10_m9/polyupd10_res_beta_0.5__64"]')
    parser.add_argument('--sm-path', type=str, default='./models/psm_upd/0123456789_64dsm_dsm_res_',
                        help='multimodal model path [default: "./models/psm_upd/0123456789_64dsm_dsm_res_"]')
    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')

    
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.allmods, args.savefolder, [args.p0_path, args.p1_path, args.p2_path, args.p3_path, args.p4_path, args.p5_path, args.p6_path, args.p7_path, args.p8_path, args.p9_path], args.sm_type, args.sm_path, args.pclf_path, args.unq_name, args.drop_p)