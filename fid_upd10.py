import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np
import os
from torchvision.utils import save_image

from polymnist_dataset import test_dataset_upd10_32x32
from h_vae_model_copy import ResVAE, ResAE
from unet_model import Unet
from polymnist_model import PMCLF
from lat_sm2_model import LSMPoly64_sm
from mopoe_model import MOPOEPolyRes, MMVAEPolyRes, MVPolyRes, MVTPolyRes
from mmplus_model import PolyMNIST_10modalities

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size):
    paired_test_dataset = test_dataset_upd10_32x32()
    # train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return test_dataloader

def gen_mix(mix_vae, predicted_mods, all_mods, p, amount, device, size_z=64, use_prod=False):
    with torch.no_grad():
        mix_vae.eval()
        p_out = {}

        if len(predicted_mods) != len(all_mods):
            present_mod = ''.join([i for i in all_mods if i not in predicted_mods])
            present_mod_list = [int(m) - int(all_mods[0]) for m in present_mod]
            if use_prod:
                all_predicted = mix_vae.cond_gen(present_mod_list, [p[mod] for mod in all_mods], use_prod=True)
            else:
                all_predicted = mix_vae.cond_gen(present_mod_list, [p[mod] for mod in all_mods])
        else:
            if isinstance(mix_vae, PolyMNIST_10modalities):
                all_predicted = mix_vae.unc_gen(p['0'].shape[0])
            else:
                z = torch.normal(mean=0, std=1, size=(p['0'].shape[0],size_z), device=device)
                all_predicted = mix_vae.sample(z)
        
        for mod in predicted_mods:
            p_out[mod] = all_predicted[int(mod) - int(all_mods[0])]
        
        if len(predicted_mods) == 1:
            return p_out[predicted_mods][:amount]
        return torch.cat([p_out[mod] for mod in predicted_mods], dim=2)

def gen_ae(score_ae, pae_dict, predicted_mods, all_mods, p, amount, device, size_z):
    with torch.no_grad():
        score_ae.eval()
        for model in pae_dict.values():
            model.eval()
        p_out = {}
        sigmas = torch.tensor(np.linspace(5, 0.1, 200)).to(device)
        
        if len(predicted_mods) == 10:
            er = 0.01
            c = 0.7
            iter_num = 2
        else:
            er = 2e-3
            c = 0.5
            iter_num = 20
        
        z = {}
        b_size = p[all_mods[0]].shape[0]

        for key in sorted(pae_dict.keys()):
            if len(predicted_mods) == 0:
                z[key] = torch.normal(mean=0, std=1, size=(p[key].shape[0],size_z), device=device)
            else:
                if key in predicted_mods:
                    z[key] = torch.normal(mean=0, std=1, size=(p[key].shape[0],size_z), device=device)
                if key not in predicted_mods:
                    z[key] = pae_dict[key].encoder(p[key])

        for s_in, s in enumerate(sigmas):
            sigma_index = torch.tensor([s_in]*b_size).to(device)
            cur_sigmas = sigmas[sigma_index].float().to(device) 
            alpha = er * (sigmas[s_in]**2)/(sigmas[-1]**2)

            # if mod in given:
            #     noised[mod] = s * torch.randn_like(z[mod])
            #     z[mod] = z[mod] + noised[mod]
            
            for i in range(iter_num):
                z_in = torch.cat([z[mod].unsqueeze(1) for mod in sorted(pae_dict.keys())], dim=1).view(-1,len(pae_dict.keys()),8,8).detach()
                sm_out = score_ae(z_in, sigma_index) / cur_sigmas.view(z_in.shape[0],*([1]*len(z_in.shape[1:])))

                for ind,mod in enumerate(sorted(pae_dict.keys())):
                    if mod in predicted_mods:
                        z[mod] = z[mod] + (alpha * sm_out[:,ind].view(b_size,size_z)) + c*torch.sqrt(2*alpha)*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
            
            # if mod in given:
            #     z[mod] = z[mod] - noised[mod]
            
        for mod in predicted_mods:
            p_out[mod] = pae_dict[mod].decoder(z[mod])
        
        if len(predicted_mods) == 1:
            return p_out[predicted_mods][:amount]
        return torch.cat([p_out[mod] for mod in predicted_mods], dim=2)


def save_sample_cond(test_loader, vae_dict, pae_dict, sm_model, score_ae, mopoe_model, mmvae_model, mvae_model, mvt_model, mmplus_model, predicted_mods, all_mods, lr1, lr2, n_comp, size_z, amount, device,  use_mean=False, schedule=False):
    with torch.no_grad():
        for vae in vae_dict.values():
            vae.eval()
        cond_accuracies = {}
        for pred in predicted_mods:
            cond_accuracies[pred] = 0
        p = {}
        p_out = {}

        images, target = next(iter(test_loader))
        # target = target.to(device)

        for key in sorted(vae_dict.keys()):
            p[key] = []
            
        for key in sorted(vae_dict.keys()):
            for digit in range(10):
                sel_target_ind = target == digit
                p[key].append(images['m'+key][sel_target_ind][:1].to(device))
            p[key] = torch.cat(p[key], dim=0)[:amount]
        
        z = {}
        out_dsm, out_ae, out_mopoe, out_mmvae, out_mvae, out_mvt, out_mmplus = [], [], [], [], [], [], []
        if len(predicted_mods) == 1:
            len_k = 5
        else:
            len_k = 1
        
        for k in range(len_k):
            for key in sorted(vae_dict.keys()):
                if len(predicted_mods) == 0:
                    z[key] = torch.normal(mean=0, std=1, size=(p[key].shape[0],size_z), device=device)
                else:
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

            # pic_tensor = []
            for mod in predicted_mods:
                p_out[mod] = vae_dict[mod].decoder(z[mod])
                # pic_tensor.append(p_out[mod][:10])
            if len(predicted_mods) == 1:
                out_dsm.append(p_out[mod][:amount])

        if len(predicted_mods) == 1:
            for i in range(len_k):
                out_dsm[i] = torch.cat([out_dsm[i][j] for j in range(amount)], dim=1)
            outs = torch.cat([out_dsm[i] for i in range(len_k)], dim=2)
            save_image(outs, './samples/new_temp_samples/pupd_dsm_'+ predicted_mods + '.png')

            for k in range(len_k):
                out_ae.append(gen_ae(score_ae, pae_dict, predicted_mods, all_mods, p, amount, device, size_z))
                out_mopoe.append(gen_mix(mopoe_model, predicted_mods, all_mods, p, amount, device))
                out_mmvae.append(gen_mix(mmvae_model, predicted_mods, all_mods, p, amount, device))
                out_mvae.append(gen_mix(mvae_model, predicted_mods, all_mods, p, amount, device))
                out_mvt.append(gen_mix(mvt_model, predicted_mods, all_mods, p, amount, device))
                out_mmplus.append(gen_mix(mmplus_model, predicted_mods, all_mods, p, amount, device))

                if k == len_k - 1:
                    for i in range(len_k):
                        out_ae[i] = torch.cat([out_ae[i][j] for j in range(amount)], dim=1)
                    outs_ae = torch.cat([out_ae[i] for i in range(len_k)], dim=2)
                    save_image(outs_ae, './samples/new_temp_samples/pupd_ae_'+ predicted_mods + '.png')

                    for i in range(len_k):
                        out_mopoe[i] = torch.cat([out_mopoe[i][j] for j in range(amount)], dim=1)
                    outs_mopoe = torch.cat([out_mopoe[i] for i in range(len_k)], dim=2)
                    save_image(outs_mopoe, './samples/new_temp_samples/pupd_mopoe_'+ predicted_mods + '.png')

                    for i in range(len_k):
                        out_mmvae[i] = torch.cat([out_mmvae[i][j] for j in range(amount)], dim=1)
                    outs_mmvae = torch.cat([out_mmvae[i] for i in range(len_k)], dim=2)
                    save_image(outs_mmvae, './samples/new_temp_samples/pupd_mmvae_'+ predicted_mods + '.png')

                    for i in range(len_k):
                        out_mvae[i] = torch.cat([out_mvae[i][j] for j in range(amount)], dim=1)
                    outs_mvae = torch.cat([out_mvae[i] for i in range(len_k)], dim=2)
                    save_image(outs_mvae, './samples/new_temp_samples/pupd_mvae_'+ predicted_mods + '.png')

                    for i in range(len_k):
                        out_mvt[i] = torch.cat([out_mvt[i][j] for j in range(amount)], dim=1)
                    outs_mvt = torch.cat([out_mvt[i] for i in range(len_k)], dim=2)
                    save_image(outs_mvt, './samples/new_temp_samples/pupd_mvt_'+ predicted_mods + '.png')

                    for i in range(len_k):
                        out_mmplus[i] = torch.cat([out_mmplus[i][j] for j in range(amount)], dim=1)
                    outs_mmplus = torch.cat([out_mmplus[i] for i in range(len_k)], dim=2)
                    save_image(outs_mmplus, './samples/new_temp_samples/pupd_mmplus_'+ predicted_mods + '.png')
            
            print("DONE", k+1, flush=True)
        else:
            out_dsm = torch.cat([p_out[mod] for mod in predicted_mods], dim=2)
            out_dsm = torchvision.utils.make_grid(out_dsm, nrow=10)
            save_image(out_dsm, './samples/new_temp_samples/pupd_dsm_'+ predicted_mods + '.png')

            out_ae = gen_ae(score_ae, pae_dict, predicted_mods, all_mods, p, amount, device, size_z)
            out_ae = torchvision.utils.make_grid(out_ae, nrow=10)
            save_image(out_ae, './samples/new_temp_samples/pupd_ae_' + predicted_mods + '.png')

            out_mopoe = gen_mix(mopoe_model, predicted_mods, all_mods, p, amount, device)
            out_mopoe = torchvision.utils.make_grid(out_mopoe, nrow=10)
            save_image(out_mopoe, './samples/new_temp_samples/pupd_mopoe_'+ predicted_mods + '.png')

            out_mmvae = gen_mix(mmvae_model, predicted_mods, all_mods, p, amount, device)
            out_mmvae = torchvision.utils.make_grid(out_mmvae, nrow=10)
            save_image(out_mmvae, './samples/new_temp_samples/pupd_mmvae_'+ predicted_mods + '.png')

            out_mvae = gen_mix(mvae_model, predicted_mods, all_mods, p, amount, device)
            out_mvae = torchvision.utils.make_grid(out_mvae, nrow=10)
            save_image(out_mvae, './samples/new_temp_samples/pupd_mvae_'+ predicted_mods + '.png')

            out_mvt = gen_mix(mvt_model, predicted_mods, all_mods, p, amount, device)
            out_mvt = torchvision.utils.make_grid(out_mvt, nrow=10)
            save_image(out_mvt, './samples/new_temp_samples/pupd_mvt_'+ predicted_mods + '.png')

            out_mmplus = gen_mix(mmplus_model, predicted_mods, all_mods, p, amount, device)
            out_mmplus = torchvision.utils.make_grid(out_mmplus, nrow=10)
            save_image(out_mmplus, './samples/new_temp_samples/pupd_mmplus_'+ predicted_mods + '.png')
            print("DONE", flush=True)
        return


def calc_poly_cond(test_loader, sample_path, vae_dict, sm_model, predicted_mods, all_mods, p_clf, lr1, lr2, n_comp, size_z, device, write_input, use_mean=False, schedule=False):
    with torch.no_grad():
        for vae in vae_dict.values():
            vae.eval()
        cond_accuracies = {}
        for pred in predicted_mods:
            cond_accuracies[pred] = 0
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in sorted(vae_dict.keys()):
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            z = {}

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
                
                if write_input:
                    save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['cond_p' + str(len(all_mods)) + '_' + mod + '_' + ''.join([i for i in all_mods if i not in predicted_mods])] + str(batch_idx) + '_')

        for mod in cond_accuracies.keys():
            cond_accuracies[mod] /= len(test_loader)
        print("Cond Coherence: ", cond_accuracies, flush=True)
        return cond_accuracies

def calc_poly_uncond(test_loader, sample_path, vae_dict, sm_model, p_clf, lr1, lr2, n_comp, size_z, device, write_input, schedule=False):
    with torch.no_grad():
        for vae in vae_dict.values():
            vae.eval()
        unc_accuracies = [0]*6
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in sorted(vae_dict.keys()):
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            z = {}
            predicted_out = {}
            for pred in sorted(vae_dict.keys()):
                z[pred] = torch.normal(mean=0, std=1, size=(p[pred].shape[0],size_z), requires_grad=True, device=device)

            for i in range(n_comp):
                z_in = torch.cat([z[mod] for mod in sorted(vae_dict.keys())], dim=1)
                sm_out = sm_model(z_in)

                for ind,mod in enumerate(sorted(vae_dict.keys())):
                    if not schedule:
                        z[mod] = z[mod] + (lr1 * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
                    else:
                        z[mod] = z[mod] + (lr1 * ((i+1)/n_comp) * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)
                        if i == n_comp - 1:
                            z[mod] = z[mod] + (1 * ((i+1)/n_comp) * sm_out[:,ind*size_z:ind*size_z+size_z]) + lr2*torch.normal(mean=0, std=1, size=z[mod].shape, device=device)


            for mod in sorted(vae_dict.keys()):
                p_out[mod] = vae_dict[mod].decoder(z[mod])
                predicted_out[mod] = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
                predicted_out[mod] = torch.argmax(predicted_out[mod], 1)
                
                if write_input:
                    save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['unc_p' + str(len(sorted(vae_dict.keys()))) + '_' + mod] + mod + '_out_' + str(batch_idx) + '_')

                
            for ind, num_eq_check in enumerate(range(5,len(list(sorted(vae_dict.keys())))+1)):
                equality_mask = (torch.stack([predicted_out[out] for out in sorted(vae_dict.keys())], dim=0) == predicted_out[list(sorted(vae_dict.keys()))[0]]).sum(dim=0)
                equality_mask = equality_mask >= num_eq_check
                unc_acc = torch.sum(equality_mask).item()
                unc_acc = unc_acc / p[list(sorted(vae_dict.keys()))[0]].shape[0]
                unc_accuracies[ind] += unc_acc
                
        for i in range(len(unc_accuracies)):
            unc_accuracies[i] =  unc_accuracies[i] / len(test_loader)
        print("UNC acc: " , unc_accuracies, flush=True)
        return unc_accuracies[-1]

def check_file_len(path, amount):
    for dir in path:
        initial_count = 0
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                initial_count += 1
        if (initial_count != amount):
            print('file len error: ', dir, flush=True)
            return False
    return True

def run(batch_size, size_z, all_mod, predicted_mod, model_paths, model_paths_ae, sm_path, score_ae_path, sm_type, pclf_path, mopoe_path, mmvae_path, mvae_path, mvt_path, mmplus_path, lr1, lr2, unc_lr1, unc_lr2, n_comp, fid_n_times, unq_name, incremental_calc):
    print('vars: ', all_mod, predicted_mod, batch_size, size_z, lr1, lr2, unc_lr1, unc_lr2, fid_n_times, sm_type, sm_path, flush=True)
    print("All mods: ", all_mod, flush=True)
    print("Predicted mod: ", predicted_mod, flush=True)
    fid_scores_cond = []
    fid_scores_unc = []
    unc_accs = []
    cond_accs = []

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:3" if cuda else "cpu")

    enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
    dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
    size_in = 32
    img_ch = 3
    pvae_dict = {}
    n_mod = len(all_mod)
    
    for ind, model_path in enumerate(model_paths):
        if str(ind) in all_mod:
            pmvae = ResVAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
            pmvae.load_state_dict(torch.load(model_path)['model_state_dict'])
            pmvae = pmvae.to(device)
            pvae_dict[str(ind)] = pmvae
    
    
    score_model = LSMPoly64_sm(n_mod, size_z)
    # print('nn sz', n_mod, size_z, flush=True)
    score_model.load_state_dict(torch.load(sm_path)['model_state_dict'])
    score_model = score_model.to(device)

    pae_dict = {}
    for ind, model_path in enumerate(model_paths_ae):
        if str(ind) in all_mod:
            pmvae = ResAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
            pmvae.load_state_dict(torch.load(model_path)['model_state_dict'])
            pmvae = pmvae.to(device)
            pae_dict[str(ind)] = pmvae

    if n_mod > 5:
        dim = 64
    else:
        dim = 32
    score_ae = Unet(dim=dim, channels=n_mod, dim_mults=(1,2,2,2), with_time_emb=True)
    score_ae.load_state_dict(torch.load(score_ae_path)['model_state_dict'])
    score_ae = score_ae.to(device)

    mopoe_vae = MOPOEPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
    mopoe_vae.load_state_dict(torch.load(mopoe_path)['model_state_dict'])
    mopoe_vae = mopoe_vae.to(device)
    mopoe_vae.eval()
        
    mmvae_vae = MMVAEPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
    mmvae_vae.load_state_dict(torch.load(mmvae_path)['model_state_dict'])
    mmvae_vae = mmvae_vae.to(device)
    mmvae_vae.eval()

    mvae = MVPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
    mvae.load_state_dict(torch.load(mvae_path)['model_state_dict'])
    mvae = mvae.to(device)
    mvae.eval()

    mvt_vae = MVTPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
    mvt_vae.load_state_dict(torch.load(mvt_path)['model_state_dict'])
    mvt_vae = mvt_vae.to(device)
    mvt_vae.eval()

    b_size = batch_size
    class Params():
        latent_dim_w = size_z // 2
        latent_dim_z = size_z // 2
        model = 'polymnist_10modalities'
        obj = 'dreg'
        K = 1
        batch_size = b_size
        epochs = 200
        beta = 0.5
        learn_prior_w_polymnist = True
        variant = 'mmvaeplus'
        tmpdir = '/tmp/'
        no_cuda = False
        n_mod = len(all_mod)
    params = Params()
    mmplus_vae = PolyMNIST_10modalities(params)
    mmplus_vae.load_state_dict(torch.load(mmplus_path)['model_state_dict'])
    mmplus_vae = mmplus_vae.to(device)
    mmplus_vae.eval()

    
    print('Models loaded', flush=True)
    
    test_dataloader = get_train_test_dataloader_upd10_32x32(batch_size)
    print('data loaded', flush=True)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()

    # save_sample_cond(test_dataloader, pvae_dict, pae_dict, score_model, score_ae, mopoe_vae, mmvae_vae, mvae, mvt_vae, mmplus_vae, '2', all_mod, lr1, lr2, n_comp, size_z, 10, device)
    # save_sample_cond(test_dataloader, pvae_dict, pae_dict, score_model, score_ae, mopoe_vae, mmvae_vae, mvae, mvt_vae, mmplus_vae, '0', all_mod, lr1, lr2, n_comp, size_z, 10, device)
    # save_sample_cond(test_dataloader, pvae_dict, pae_dict, score_model, score_ae, mopoe_vae, mmvae_vae, mvae, mvt_vae, mmplus_vae, '5', all_mod, lr1, lr2, n_comp, size_z, 10, device)
    # # save_sample_cond(test_dataloader, pvae_dict, pae_dict, score_model, score_ae, mopoe_vae, mmvae_vae, mvae, mvt_vae, mmplus_vae, '0123456789', all_mod, unc_lr1, unc_lr2, n_comp, size_z, 10, device)
    # return
    # O
    sample_input_path = []
    sample_output_path = []

    sample_path = {}
    for mod in all_mod:
        sample_path['p' + mod] = './samples/p' + mod + '/'
        sample_input_path.append('./samples/p' + mod + '/')
        if incremental_calc == 0:
            if len(predicted_mod) == 0:
                sample_path['unc_p' + str(len(all_mod)) + '_' + mod] = './samples/unc_p' + str(len(all_mod)) + '_' + mod + '/'
                sample_output_path.append('./samples/unc_p' + str(len(all_mod)) + '_' + mod + '/')
            else:
                for pred in predicted_mod:
                    sample_path['cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])] = './samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/'
                    if './samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/' not in sample_output_path:
                        sample_output_path.append('./samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/')
    
    for p in sample_input_path + sample_output_path:
        if not os.path.exists(p):
            os.makedirs(p)
    
    print("Input path: ", sample_input_path, flush=True)
    print("Output path: ", sample_output_path, flush=True)

    write_input = False
    # num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = 2

    if incremental_calc:

        calculated_mod = '9'
        print('incremental evaluation ', calculated_mod)
        all_fid_score, all_cond_acc = [], []

        for g in range(9):
            predicted_mod = all_mod[g+1:len(all_mod)]
            print('my pred: ', predicted_mod, flush=True)
            
            for pred in predicted_mod:
                sample_path['cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])] = './samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/'
                if './samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/' not in sample_output_path:
                    sample_output_path.append('./samples/cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/')
            
            for p in sample_input_path + sample_output_path:
                if not os.path.exists(p):
                    os.makedirs(p)

            if len(predicted_mod) > 0:
                fid_scores_cond, cond_accs = [], []
                for i in range(fid_n_times):
                    # fid_scores_cond.append([])
                    if not check_file_len(sample_input_path, 10000):
                        write_input = True
                    # write_input = True

                    cond_coherence = calc_poly_cond(test_dataloader, sample_path, pvae_dict, score_model, predicted_mod, all_mod, poly_clf, lr1, lr2, n_comp, size_z, device, write_input)
                    
                    if not check_file_len(sample_input_path + sample_output_path, 10000):
                        raise Exception('file len check not correct!')

                    for pred in calculated_mod:
                        cond_p = calculate_fid_given_paths([sample_path['p'+pred], sample_path['cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])]], batch_size, device, 2048, num_workers)
                        fid_scores_cond.append(cond_p)
                    cond_accs.append(cond_coherence[calculated_mod])
            
            # fid_scores_cond = np.array(fid_scores_cond)
            # cond_accs = np.array(cond_accs)
            
            all_fid_score.append(fid_scores_cond)
            all_cond_acc.append(cond_accs)
            
            print("Conditional coherence: ", np.array(cond_accs).mean(), flush=True)
            print("Mean Fid Scores conditional: ", np.array(fid_scores_cond).mean(), flush=True)

        np.save('./ar/dsm_increm_fid_' + unq_name + calculated_mod, np.array(all_fid_score))
        np.save('./ar/dsm_increm_acc_' + unq_name + calculated_mod, np.array(all_cond_acc))
        return
    
    else:
        if len(predicted_mod) > 0:
            print('conditional evaluation')
            for i in range(fid_n_times):
                fid_scores_cond.append([])
                # if not check_file_len(sample_input_path, 10000):
                #     write_input = True
                write_input = True

                cond_coherence = calc_poly_cond(test_dataloader, sample_path, pvae_dict, score_model, predicted_mod, all_mod, poly_clf, lr1, lr2, n_comp, size_z, device, write_input)
                
                if not check_file_len(sample_input_path + sample_output_path, 10000):
                    raise Exception('file len check not correct!')

                for pred in predicted_mod:
                    cond_p = calculate_fid_given_paths([sample_path['p'+pred], sample_path['cond_p' + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])]], batch_size, device, 2048, num_workers)
                    fid_scores_cond[i].append(cond_p)
                print('fids: ', fid_scores_cond[i], flush=True)
                cond_accs.append(list(cond_coherence.values()))
            
            fid_scores_cond = np.array(fid_scores_cond)
            cond_accs = np.array(cond_accs)
            
            print("Conditional coherence: ", np.mean(cond_accs, axis=0), flush=True)
            print("Mean Fid Scores conditional: ", np.mean(fid_scores_cond, axis=0), flush=True)

            np.save('./ar/cond_fid_dsm_' + unq_name + predicted_mod, np.array(fid_scores_cond))
            np.save('./ar/cond_acc_dsm_' + unq_name + predicted_mod, np.array(cond_accs))
            return

        print(" ", flush=True)
    
        if len(predicted_mod) == 0:
            print('unconditional evaluation')
            for i in range(fid_n_times):
                fid_scores_unc.append([])
                if not check_file_len(sample_input_path, 10000):
                    write_input = True
                # write_input = False

                unc_coherence = calc_poly_uncond(test_dataloader, sample_path, pvae_dict, score_model, poly_clf, unc_lr1, unc_lr2, n_comp, size_z, device, write_input)
                
                if not check_file_len(sample_input_path + sample_output_path, 10000):
                    raise Exception('file len check not correct!')
                
                for mod in all_mod:
                    unc_p = calculate_fid_given_paths([sample_path['p'+mod], sample_path['unc_p' + str(len(all_mod)) + '_' + mod]], batch_size, device, 2048, num_workers)
                    fid_scores_unc[i].append(unc_p)
                unc_accs.append(unc_coherence)
        
            fid_scores_unc = np.array(fid_scores_unc)
            unc_accs = np.array(unc_accs)

            print("Mean Fid Scores unconditional: ", np.mean(fid_scores_unc, axis=0), flush=True)
            print("Unc coherence: ", np.mean(unc_accs), flush=True)

            np.save('./ar/dsm_unc_fid_' + unq_name + all_mod, np.array(fid_scores_unc))
            np.save('./ar/dsm_unc_acc_' + unq_name + all_mod, np.array(unc_accs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--all_mod', type=str, default='0123456789',
                        help='all modalities [default: "0123456789"]')
    parser.add_argument('--predicted_mod', type=str, default='',
                        help='predicted modalities [default: ""]')
    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--fid-n-times', type=int, default=5,
                        help='number of times to repeat fid calc [default: 5]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--lr1', type=float, default=0.2,
                        help='lr1 [default: 0.2]')
    parser.add_argument('--lr2', type=float, default=0.02,
                        help='lr2 [default: 0.02]')
    parser.add_argument('--unc-lr1', type=float, default=0.1,
                        help='unc-lr1 [default: 0.1]')
    parser.add_argument('--unc-lr2', type=float, default=0.15,
                        help='unc-lr2 [default: 0.15]')
    parser.add_argument('--ncomp', type=int, default=40,
                        help='size of LD iterations [default: 40]')
    parser.add_argument('--sm-type', type=str, default='dsm',
                        help='loss type [default: "dsm"]')

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
    # parser.add_argument('--sm-path', type=str, default='./models/psm_upd/psm_upd_res_64_dsm_res___',
    #                     help='multimodal model path [default: "./models/psm_upd/psm_upd_res_64_dsm_res___"]')
    parser.add_argument('--sm-path', type=str, default='./models/psm_upd/0123456789_64dsm_dsm_res_',
                        help='multimodal model path [default: "./models/psm_upd/0123456789_64dsm_dsm_res_"]')

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
    parser.add_argument('--score-ae', type=str, default='./models/psm_upd/0123456789_64_AE_psm_aeNEWre5_dim32_s5_01_200_dim64_unet_',
                        help='score ae model path [default: "./models/psm_upd/0123456789_64_AE_psm_aeNEWre5_dim32_s5_01_200_dim64_unet_"]')

    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')

    parser.add_argument('--mopoe-path', type=str, default='./models/mopoe_pupd/mopoe_pupd_perm_vae_res_beta_0.5__640.001',
                        help='mopoe model path [default: "./models/mopoe_pupd/mopoe_pupd_perm_vae_res_beta_0.5__640.001"]')
    parser.add_argument('--mmvae-path', type=str, default='./models/mopoe_pupd/mmvae_vae_res_beta_0.5__640.001',
                        help='mmvae model path [default: "./models/mopoe_pupd/mmvae_vae_res_beta_0.5__640.001"]')
    parser.add_argument('--mvae-path', type=str, default='./models/mopoe_pupd/mvae_vae_res_beta_0.5__640.001',
                        help='mvae model path [default: "./models/mopoe_pupd/mvae_vae_res_beta_0.5__640.001"]')
    parser.add_argument('--mvt-path', type=str, default='./models/mopoe_pupd/mvt_vae_res_beta_0.5__640123456789_',
                        help='mvtcae model path [default: "./models/mopoe_pupd/mvt_vae_res_beta_0.5__640123456789_"]')
    parser.add_argument('--mmplus-path', type=str, default='./models/mopoe_pupd/mmplusNEW_vae_res_beta_0.5__640123456789__k_1dreg',
                        help='mmplus model path [default: "./models/mopoe_pupd/mmplusNEW_vae_res_beta_0.5__640123456789__k_1dreg"]')

    parser.add_argument('--unq-name', type=str, default='',
                        help='unique name for experiment [default: ""]')
    parser.add_argument('--inc-fid', type=int, default=1,
                        help='calculate fid incrementally [default: 1]')

    args = parser.parse_args()
    model_paths = [args.p0_path, args.p1_path, args.p2_path, args.p3_path, args.p4_path, args.p5_path, args.p6_path, args.p7_path, args.p8_path, args.p9_path]
    model_paths_ae = [args.p0_path_ae, args.p1_path_ae, args.p2_path_ae, args.p3_path_ae, args.p4_path_ae, args.p5_path_ae, args.p6_path_ae, args.p7_path_ae, args.p8_path_ae, args.p9_path_ae]

    run(args.batch_size, args.size_z, args.all_mod, args.predicted_mod, model_paths, model_paths_ae, args.sm_path, args.score_ae, args.sm_type, args.pclf_path, \
        args.mopoe_path, args.mmvae_path, args.mvae_path, args.mvt_path, args.mmplus_path, \
        args.lr1, args.lr2, args.unc_lr1, args.unc_lr2, args.ncomp, args.fid_n_times, args.unq_name, args.inc_fid)


