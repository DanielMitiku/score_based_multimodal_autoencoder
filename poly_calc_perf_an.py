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
import os

from polymnist_dataset import test_dataset_upd10_32x32, test_dataset_upd10_28x28
from polymnist_model import PMCLF
from mopoe_model import MOPOEPolyRes, MMVAEPolyRes, MVPolyRes, MVTPolyRes
from mmplus_model_orig import *

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
import shutil

def get_train_test_dataloader_upd10(batch_size, mix_type='mopoe', test=True):
    print("Test Dataset " if test else "Val Dataset ", flush=True)
    if mix_type == 'mopoe':
        drop_last = True
    else:
        drop_last = False
    if mix_type == 'mmplus' or mix_type == 'mmplusOrig':
        paired_test_dataset = test_dataset_upd10_28x28(test=test)
    else:
        paired_test_dataset = test_dataset_upd10_32x32(test=test)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=drop_last)
    return test_dataloader

def gen_mix(mix_vae, mix_type, predicted_mods, all_mods, p, amount, device, size_z=64, use_prod=False):
    with torch.no_grad():
        if mix_type in ['mopoe', 'mvt', 'mvae', 'mmvae']:
            size_z = size_z  * len(all_mods)
        mix_vae.eval()
        p_out = {}

        if len(predicted_mods) != len(all_mods):
            present_mod = ''.join([i for i in all_mods if i not in predicted_mods])
            present_mod_list = [int(m) - int(all_mods[0]) for m in present_mod]
            if use_prod:
                all_predicted = mix_vae.cond_gen(present_mod_list, [p[mod] for mod in all_mods], use_prod=use_prod)
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
    
def save_sample_cond(test_loader, mix_vae, mix_type, predicted_mods, all_mods, amount, device):
    with torch.no_grad():
        p = {}
        p_out = {}

        images, target = next(iter(test_loader))

        for key in all_mods:
            p[key] = []
            
        for key in all_mods:
            for digit in range(10):
                sel_target_ind = target == digit
                p[key].append(images['m'+key][sel_target_ind][:1].to(device))
            p[key] = torch.cat(p[key], dim=0)[:amount]

        out_mix = []
        if len(predicted_mods) == 1:
            len_k = 5
        else:
            len_k = 1

        if len(predicted_mods) == 1:
            for k in range(len_k):
                out_mix.append(gen_mix(mix_vae, mix_type, predicted_mods, all_mods, p, amount, device))

                if k == len_k - 1:
                    for i in range(len_k):
                        out_mix[i] = torch.cat([out_mix[i][j] for j in range(amount)], dim=1)
                    outs_mix = torch.cat([out_mix[i] for i in range(len_k)], dim=2)
                    save_image(outs_mix, './AN_samples/'+ mix_type + '_' + all_mods + '_' + predicted_mods + '.png')

            print("DONE", k+1, flush=True)
        else:
            out_mix = gen_mix(mix_vae, mix_type, predicted_mods, all_mods, p, amount, device)
            out_mix = torchvision.utils.make_grid(out_mix, nrow=10)
            save_image(out_mix, './AN_samples/'+ mix_type + '_' + all_mods + '_' + predicted_mods + '.png')
            print("DONE", flush=True)
        return

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
                if mix_type == 'mmplus' or mix_type == 'mmplusOrig':
                    predicted_out = p_clf(p_out[mod].view(-1,3,28,28))
                else:
                    predicted_out = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
                predicted_out = torch.argmax(predicted_out, 1)
                
                cond_acc = torch.sum(predicted_out == target).item()
                cond_acc = cond_acc / p[mod].shape[0]
                cond_accuracies[mod] += cond_acc
                
                save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + mod + '_' + ''.join([i for i in all_mods if i not in predicted_mods])] + str(batch_idx) + '_')

        for mod in cond_accuracies.keys():
            cond_accuracies[mod] /= len(test_loader)
        print("-- Cond Coherence: ", cond_accuracies, flush=True)
        
        fid_scores_cond = {}
        for pred in predicted_mods:
            fid_p = calculate_fid_given_paths([sample_path['p'+pred], sample_path['cond_p' + mix_type + str(len(all_mods)) + '_' + pred + '_' + ''.join([i for i in all_mods if i not in predicted_mods])]], 256, device, 2048, 2)
            fid_scores_cond[pred] = fid_p
        print("-- Cond Fid: ", fid_scores_cond, flush=True)

        return cond_accuracies, fid_scores_cond

def calc_poly_uncond(test_loader, sample_path, mix_vae, mix_type, all_mods, p_clf, size_z, device, write_input=True):
    with torch.no_grad():
        mix_vae.eval()
        unc_accuracies = [0]*(len(all_mods)-1)
        p = {}
        p_out = {}

        for batch_idx, (images, target) in enumerate(test_loader):
            for key in all_mods:
                p[key] = images['m'+key].to(device)
            target = target.to(device)

            z = torch.normal(mean=0, std=1, size=(p[all_mods[0]].shape[0],size_z), device=device)
            if mix_type == 'mmplus' or mix_type == 'mmplusOrig':
                outs = mix_vae.unc_gen(p[all_mods[0]].shape[0])
            else:
                outs = mix_vae.sample(z)
            predicted_out = {}

            for ind, mod in enumerate(all_mods):
                p_out[mod] = outs[ind]
                if mix_type == 'mmplus' or mix_type == 'mmplusOrig':
                    predicted_out[mod] = p_clf(p_out[mod].view(-1,3,28,28))
                else:
                    predicted_out[mod] = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
                predicted_out[mod] = torch.argmax(predicted_out[mod], 1)
                
                save_batch_image(p[mod], sample_path['p' + mod] + 'p' + mod + str(batch_idx) + '_')
                save_batch_image(p_out[mod], sample_path['unc_p' + mix_type + str(len(all_mods)) + '_' + mod] + mod + '_out_' + str(batch_idx) + '_')
                
            for ind, num_eq_check in enumerate(range(2,len(all_mods)+1)):
                equality_mask = (torch.stack([predicted_out[out] for out in all_mods], dim=0) == predicted_out[all_mods[0]]).sum(dim=0)
                equality_mask = equality_mask >= num_eq_check
                unc_acc = torch.sum(equality_mask).item()
                unc_acc = unc_acc / p[all_mods[0]].shape[0]
                unc_accuracies[ind] += unc_acc
                
        for i in range(len(unc_accuracies)):
            unc_accuracies[i] =  unc_accuracies[i] / len(test_loader)
        print("-- UNC acc: " , unc_accuracies, flush=True)

        fid_scores_unc = {}
        for mod in all_mods:
            unc_p = calculate_fid_given_paths([sample_path['p'+mod], sample_path['unc_p' + mix_type + str(len(all_mods)) + '_' + mod]], 256, device, 2048, 2)
            fid_scores_unc[mod] = unc_p
        
        print("-- UNC fid: " , fid_scores_unc, flush=True)

        return unc_accuracies, fid_scores_unc

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

def run(batch_size, size_z, all_mod, predicted_mod, mix_type, mix_path, pclf_path, fid_n_times, unq_name, cuda_num, incremental_fid, inc_calc_mod, test_set):
    print('vars: ', all_mod, predicted_mod, batch_size, size_z, fid_n_times, mix_type, flush=True)
    print('Incremental FID = ', 'True' if incremental_fid else 'False')
    print("All mods: ", all_mod, flush=True)
    print("Predicted mod: ", predicted_mod, flush=True)
    print("mix path: ", mix_path, flush=True)
    fid_scores_cond = []
    fid_scores_unc = []
    unc_accs = []
    cond_accs = []

    unq_name += all_mod

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, cuda_num,  flush=True)
    device = torch.device("cuda:" + str(cuda_num) if cuda else "cpu")

    test_set = True if test_set else False
    print("test set: ", test_set, flush=True)

    enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
    dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
    size_in = 32
    img_ch = 3
    n_mod = len(all_mod)

    if mix_type == 'mopoe':
        size_z = size_z * n_mod
        mix_vae = MOPOEPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        mix_vae.load_state_dict(torch.load(mix_path, map_location=device)['model_state_dict'])
        print(mix_path)
    elif mix_type == 'mmvae':
        size_z = size_z * n_mod
        mix_vae = MMVAEPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        mix_vae.load_state_dict(torch.load(mix_path, map_location=device)['model_state_dict'])
        print(mix_path)
    elif mix_type == 'mvae':
        size_z = size_z * n_mod
        mix_vae = MVPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        mix_vae.load_state_dict(torch.load(mix_path, map_location=device)['model_state_dict'])
        print(mix_path)
    elif mix_type == 'mvt':
        size_z = size_z * n_mod
        mix_vae = MVTPolyRes(len(all_mod), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        mix_vae.load_state_dict(torch.load(mix_path, map_location=device)['model_state_dict'])
        print(mix_path)
    elif mix_type == 'mmplus' or mix_type == 'mmplusOrig':
        b_size = batch_size
        # just for model initialization, no training here
        class Params():
            latent_dim_w = size_z // 2
            latent_dim_z = size_z // 2
            model = 'polymnist_10modalities'
            obj = 'iwae'
            K = 1
            batch_size = b_size
            epochs = 200
            beta = 1 
            learn_prior_w_polymnist = True
            variant = 'mmvaeplus'
            tmpdir = '/tmp/'
            no_cuda = False
            n_mod = len(all_mod)
        print(mix_path)
        params = Params()
        mix_vae = PolyMNIST_10modalities(params)
        mix_vae.load_state_dict(torch.load(mix_path, map_location=device)['model_state_dict'])

    mix_vae = mix_vae.to(device)

    test_dataloader = get_train_test_dataloader_upd10(batch_size, mix_type=mix_type, test=test_set)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()

    # save_sample_cond(test_dataloader, mix_vae, mix_type, '5', all_mod, 10, device)
    # # save_sample_cond(test_dataloader, mix_vae, mix_type, '0123456789', all_mod, 10, device)
    # return

    sample_path = {}
    temp_rand = torch.randint(1000000, size=(1,)).item()
    temp_dir_name = './samp/t_' + mix_type + all_mod + '_' + str(temp_rand) + '/'
    print('temp dir: ', temp_dir_name, flush=True)

    sample_input_path =[]
    sample_output_path = []

    for mod in all_mod:
        sample_path['p' + mod] = temp_dir_name + 'p' + mod + '/'
        sample_input_path.append(temp_dir_name + 'p' + mod + '/')

        sample_path['unc_p' + mix_type + str(len(all_mod)) + '_' + mod] = temp_dir_name + 'unc_p' + mix_type + str(len(all_mod)) + '_' + mod + '/'
        sample_output_path.append(temp_dir_name + 'unc_p' + mix_type + str(len(all_mod)) + '_' + mod + '/')

    if incremental_fid == 0:
        for pred in predicted_mod:
            sample_path['cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])] = temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/'
            sample_output_path.append(temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/')
    
    elif incremental_fid == 1:
        all_wo_inc = all_mod.replace(inc_calc_mod, '')
        for i in range(1,len(all_mod)):
            predicted_mod = ''.join(sorted(all_wo_inc[i:] + inc_calc_mod))
            for pred in predicted_mod:
                sample_path['cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])] \
                    = temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/'
                sample_output_path.append(temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/')

    elif incremental_fid == 9: # create directory structure for calculating given rest
        for mod in all_mod:
            predicted_mod = mod
            pred = mod
            sample_path['cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod])] \
                = temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/'
            sample_output_path.append(temp_dir_name + 'cond_p' + mix_type + str(len(all_mod)) + '_' + pred + '_' + ''.join([i for i in all_mod if i not in predicted_mod]) + '/')
    
    for p in sample_input_path + sample_output_path:
        if not os.path.exists(p):
            os.makedirs(p)
    
    print("Input path: ", sample_input_path, flush=True)
    print("Output path: ", sample_output_path, flush=True)

    num_workers = 2 

    scores = {}

    if incremental_fid == 9: # calc performance of model using both conditional and unconditional metrics
        print('unconditional evaluation', flush=True)
        for i in range(fid_n_times):
            average_unc_fid = 0
            average_unc_acc = 0

            unc_coherence, unc_fid = calc_poly_uncond(test_dataloader, sample_path, mix_vae, mix_type, all_mod, poly_clf, size_z, device)

            for mod in all_mod:
                average_unc_fid += unc_fid[mod]
            for k in range(len(unc_coherence)):
                average_unc_acc += unc_coherence[k]

            average_unc_fid /= len(all_mod)
            average_unc_acc /= len(unc_coherence)

            fid_scores_unc.append(average_unc_fid)
            unc_accs.append(average_unc_acc)

        fid_scores_unc = np.array(fid_scores_unc)
        unc_accs = np.array(unc_accs)

        scores['unc_acc'] = np.mean(unc_accs).item()
        scores['unc_fid'] = np.mean(fid_scores_unc).item()

        print(mix_type, " Mean Fid Scores unconditional: ", scores['unc_fid'], flush=True)
        print(mix_type, " Mean Unc coherence: ", scores['unc_acc'], flush=True)

        
        print(" ", flush=True)
        print('conditional evaluation given rest', flush=True)
        for i in range(fid_n_times):
            average_cond_fid = 0
            average_cond_acc = 0

            for mod in all_mod:
                cond_acc, cond_fid = calc_poly_cond(test_dataloader, sample_path, mix_vae, mix_type, mod, all_mod, poly_clf, size_z, device)
                average_cond_acc += cond_acc[mod]
                average_cond_fid += cond_fid[mod]
            
            average_cond_fid /= len(all_mod)
            average_cond_acc /= len(all_mod)

            fid_scores_cond.append(average_cond_fid)
            cond_accs.append(average_cond_acc)

        fid_scores_cond = np.array(fid_scores_cond)
        cond_accs = np.array(cond_accs)

        scores['cond_acc'] = np.mean(cond_accs).item()
        scores['cond_fid'] = np.mean(fid_scores_cond).item()

        print(mix_type, " Mean Fid Scores conditional: ", scores['cond_fid'], flush=True)
        print(mix_type, " Mean Cond coherence: ", scores['cond_acc'], flush=True)

        total_score = (100/scores['cond_fid']) + (100/scores['unc_fid']) + scores['cond_acc'] + scores['unc_acc']
        print(mix_type, " Total score = ", total_score, " \n", flush=True )
            

    elif incremental_fid == 1:
        print('incremental evaluation on mod ', inc_calc_mod, flush=True)
        all_fid_score, all_cond_acc = [], []
        all_wo_inc = all_mod.replace(inc_calc_mod, '')

        for i in range(1, len(all_mod)):
            predicted_mods = ''.join(sorted(all_wo_inc[i:] + inc_calc_mod))
            print("predicted mods: ", predicted_mods, flush=True)
            fid_scores_cond, cond_accs = [], []

            for k in range(fid_n_times):
                cond_acc, cond_fid = calc_poly_cond(test_dataloader, sample_path, mix_vae, mix_type, predicted_mods, all_mod, poly_clf, size_z, device)
                fid_scores_cond.append(cond_fid[inc_calc_mod])
                cond_accs.append(cond_acc[inc_calc_mod])
            print(" ")

            all_fid_score.append(fid_scores_cond)
            all_cond_acc.append(cond_accs)

        print(mix_type, " Conditional coherence: ", np.array(all_cond_acc).mean(axis=-1), flush=True)
        print(mix_type, " Mean Fid Scores conditional: ", np.array(all_fid_score).mean(axis=-1), flush=True)

        np.save('./ar_AN/increm_fid_' + mix_type + unq_name + inc_calc_mod, np.array(all_fid_score))
        np.save('./ar_AN/increm_acc_' + mix_type + unq_name + inc_calc_mod, np.array(all_cond_acc))
        return
    
    else:

        if len(predicted_mod) > 0:
            print('conditional evaluation on ', predicted_mod, flush=True)
            fid_scores_cond, cond_accs = [], []
            for i in range(fid_n_times):                
                cond_acc, cond_fid = calc_poly_cond(test_dataloader, sample_path, mix_vae, mix_type, predicted_mod, all_mod, poly_clf, size_z, device)
                fid_scores_cond.append(list(cond_fid.values()))
                cond_accs.append(list(cond_acc.values()))
            
            print(" ")
            fid_scores_cond = np.array(fid_scores_cond)
            cond_accs = np.array(cond_accs)
            
            print(mix_type, " Conditional coherence: ", np.mean(cond_accs, axis=0), flush=True)
            print(mix_type, " Mean Fid Scores conditional: ", np.mean(fid_scores_cond, axis=0), flush=True)

            np.save('./ar_AN/cond_fid_' + mix_type + unq_name + predicted_mod, np.array(fid_scores_cond))
            np.save('./ar_AN/cond_acc_' + mix_type + unq_name + predicted_mod, np.array(cond_accs))
            return

        if len(predicted_mod) == 0:
            print('unconditional evaluation', flush=True)
            fid_scores_unc, unc_accs = [], []
            for i in range(fid_n_times):
                unc_coherence, unc_fid = calc_poly_uncond(test_dataloader, sample_path, mix_vae, mix_type, all_mod, poly_clf, size_z, device)
                fid_scores_unc.append(list(unc_fid.values()))
                unc_accs.append(unc_coherence)

            fid_scores_unc = np.array(fid_scores_unc)
            unc_accs = np.array(unc_accs)

            print(mix_type, " Mean Fid Scores unconditional: ", np.mean(fid_scores_unc, axis=0), flush=True)
            print(mix_type, " Unc coherence: ", np.mean(unc_accs, axis=0), flush=True)

            np.save('./ar_AN/uncond_fid_' + mix_type + unq_name, fid_scores_unc)
            np.save('./ar_AN/uncond_acc_' + mix_type + unq_name, unc_accs)
            return

    shutil.rmtree(temp_dir_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--all_mod', type=str, default='0123456789',
                        help='all modalities [default: "0123456789"]')
    parser.add_argument('--predicted_mod', type=str, default='',
                        help='predicted modalities [default: ""]')
    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--fid-n-times', type=int, default=1,
                        help='number of times to repeat fid calc [default: 1]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='number of cuda_gpu [default: 0]')
    parser.add_argument('--inc-fid', type=int, default=9,
                        help='how to calculate fid [default: 9]')
    parser.add_argument('--inc-calc-mod', type=str, default='9',
                        help='incremental calculation mod [default: "9"]')
    parser.add_argument('--unq-name', type=str, default='',
                        help='unique name for experiment [default: ""]')
    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')
    parser.add_argument('--test-set', type=int, default=0,
                        help='if 1, test dataset else val dataset [default: 0]')
    # parser.add_argument('--mopoe-path', type=str, default='./models/mopoe/mopoeNEW_vae_res_beta_2.5__6400123456789__AN_epoch_200',
    #                     help='mopoe model path [default: "./models/mopoe/mopoeNEW_vae_res_beta_2.5__6400123456789__AN_epoch_200"]')
    # parser.add_argument('--mmvae-path', type=str, default='./models/mmvae/mmvaeNEW_vae_res_beta_2.5__6400123456789__AN_epoch_200',
    #                     help='mmvae model path [default: "./models/mmvae/mmvaeNEW_vae_res_beta_2.5__6400123456789__AN_epoch_200"]')
    # parser.add_argument('--mvae-path', type=str, default='./models/mvae/mvaeNEW_vae_res_beta_1.0__6400123456789__AN_epoch_200',
    #                     help='mvae model path [default: "./models/mvae/mvaeNEW_vae_res_beta_1.0__6400123456789__AN_epoch_200"]')
    # parser.add_argument('--mvt-path', type=str, default='./models/mvt/mvtNEW_vae_res_beta_0.1__6400123456789__AN_fid_77.4_epoch_200',
    #                     help='mvtcae model path [default: "./models/mvt/mvtNEW_vae_res_beta_0.1__6400123456789__AN_fid_77.4_epoch_200"]')
    # parser.add_argument('--mmplus-path', type=str, default='./models/mmplusOrig/mmplusOrigNEW_vae_res_beta_5.0__640123456789__k_1iwaefid_110.82087349097567_epoch_200',
    #                     help='mmplus model path [default: "./models/mmplusOrig/mmplusOrigNEW_vae_res_beta_5.0__640123456789__k_1iwaefid_110.82087349097567_epoch_200"]')

    parser.add_argument('--mix-path', type=str, default='./models/mopoe_pupd/_',
                        help='mix vae model path [default: "./models/mopoe_pupd/_"]')

    parser.add_argument('--mix-type', type=str, default='mopoe',
                        help='mixture type [default: "mopoe"]')

    args = parser.parse_args()
    run(args.batch_size, args.size_z, args.all_mod, args.predicted_mod, args.mix_type, args.mix_path, args.pclf_path, args.fid_n_times, args.unq_name, args.cuda, args.inc_fid, args.inc_calc_mod, args.test_set)


