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
import shutil

from polymnist_dataset import get_train_test_dataset_upd10_32x32
from mopoe_model import MOPOEPolyRes, MMVAEPolyRes, MVPolyRes, MVTPolyRes
from polymnist_model import PMCLF

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size, mix_type='mopoe'):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_32x32()
    if mix_type == 'mopoe':
        drop_last = True
    else:
        drop_last = False
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=drop_last)
    return train_dataloader, val_dataloader, test_dataloader

# def get_train_test_dataloader(batch_size):
#     paired_train_dataset, paired_test_dataset = get_train_test_dataset()
#     train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     return train_dataloader, test_dataloader

def calc_kl_loss(mu, logvar, cons=1):
    return cons * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[0]

def calc_kl_loss_2(mu0, logvar0, mu1, logvar1, cons=1):
    kl2 = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
    return cons * kl2 / mu0.shape[0]

def image_loss(x_hat, x, cons=1):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x) / x.shape[0]
    return cons*recon_loss

def total_rec_loss(outs, inputs):
    rec_loss = 0
    for i in range(len(outs)):
        rec_loss += image_loss(outs[i], inputs[i])
    # return (1/len(outs)) * rec_loss
    return 1 * rec_loss

def total_kl_loss(mus, logvars, cons=1):
    kl_losses = 0
    for i in range(len(mus)):
        kl_losses += calc_kl_loss(mus[i], logvars[i])
    return (1/len(mus)) * cons * kl_losses

def mvae_loss(elbo_terms, inputs, kl_cons=1, elbo_subsample=True):
    assert len(elbo_terms) == (len(inputs) + 2)
    rec, kl = 0, 0

    if elbo_subsample:
        # calc individual elbo loss
        for i in range(len(inputs)):
            elbo = elbo_terms[i]
            kl += calc_kl_loss(elbo[0], elbo[1])
            rec += image_loss(elbo[2], inputs[i])
        
        # calc kth elbo loss
        kl += calc_kl_loss(elbo_terms[len(inputs)+1][0], elbo_terms[len(inputs)+1][1])
        k_idxs = elbo_terms[len(inputs)+1][2]
        k_outs = elbo_terms[len(inputs)+1][3]
        for i, k_idx in enumerate(k_idxs):
            rec += image_loss(k_outs[i], inputs[k_idx])

    # calc joint elbo loss
    kl += calc_kl_loss(elbo_terms[len(inputs)][0], elbo_terms[len(inputs)][1])
    for i in range(len(inputs)):
        rec += image_loss(elbo_terms[len(inputs)][2][i], inputs[i])

    return rec, kl_cons * kl

def mvt_loss(elbo_terms, inputs, kl_cons=1, alpha=0.9):
    assert len(elbo_terms) == (len(inputs) + 1)

    # calc joint elbo loss
    kl_joint = calc_kl_loss(elbo_terms[len(inputs)][0], elbo_terms[len(inputs)][1])
    rec = 0
    for i in range(len(inputs)):
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


def train_model(train_loader, model, mix_type, optimizer, all_mods, device, kl_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):

        inputs = [images['m' + mod].to(device) for mod in all_mods]

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

def evaluate_model(val_loader, model, mix_type, all_mods, device, epoch, vae_type, show=True, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, target) in enumerate(val_loader):

            inputs = [images['m' + mod].to(device) for mod in all_mods]

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

            if batch_idx == 0:
                # Plot random input and output
                sample_idx = torch.randint(inputs[0].shape[0], size=(1,)).item()
                sample_in = torch.cat([input[sample_idx].permute(1,2,0) for input in inputs], dim=0)
                if mix_type == 'mvae' or mix_type == 'mvt':
                    sample_out = torch.cat([elbo_terms[len(inputs)][2][i][sample_idx].permute(1,2,0) for i in range(len(inputs))], dim=0)
                else:
                    sample_out = torch.cat([out[sample_idx].permute(1,2,0) for out in outs], dim=0)

                concat_tensor = torch.cat([sample_in, sample_out], dim=1)
                # concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('    Input   Output  ')
                plt.axis("off")
                plt.savefig('./images/' + mix_type + '/' + 'recon_' + mix_type + all_mods + "_vae_" + vae_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx) + ".png")

                present_mod_list = [int(all_mods[0])]
                all_predicted = model.cond_gen(present_mod_list, [inputs[int(mod)][:1] for mod in all_mods])
                sample_out = torch.cat([all_predicted[i][0].permute(1,2,0) for i in range(len(all_mods))], dim=0)

                present_mod_list2 = [int(all_mods[int(i)]) for i in all_mods[:len(all_mods)-1]]
                all_predicted2 = model.cond_gen(present_mod_list2, [inputs[int(mod)][:1] for mod in all_mods])
                sample_out2 = torch.cat([all_predicted2[i][0].permute(1,2,0) for i in range(len(all_mods))], dim=0)

                sample_in = torch.cat([input[0].permute(1,2,0) for input in inputs], dim=0)
                concat_tensor = torch.cat([sample_in, sample_out, sample_out2], dim=1)
                # concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('  Input  GOne   GRest  ')
                plt.axis("off")
                plt.savefig('./images/' + mix_type + '/' + 'cond_' + mix_type + all_mods + "_vae_" + vae_type + str(kl_cons) + "_" + str(epoch) + '__' + str(batch_idx) + ".png")
            
        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        # print("fids: ", fids, flush=True)
        print("Validation loss: ", losses, flush=True)
        return list(losses.values())  

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
                predicted_out = p_clf(p_out[mod].view(-1,3,32,32)[:,:,2:30,2:30])
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


def run(epochs, batch_size, lr, size_z, beta, unq_name, vae_type, mix_type, all_mods, cuda_num, pclf_path):
    kl_cons = beta
    print('vars: ', epochs, batch_size, lr, size_z, kl_cons, unq_name, vae_type, mix_type)
    train_losses, val_losses = [], []
    
    for p in ['./models/' + mix_type + '/', './plots/' + mix_type + '/', './images/' + mix_type + '/']:
        if not os.path.exists(p):
            os.makedirs(p)

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, cuda_num, flush=True)
    device = torch.device("cuda:"+str(cuda_num))
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    if vae_type == 'res':
        enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
        dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
        size_in = 32
        img_ch = 3
        if mix_type == 'mopoe':
            mix_vae = MOPOEPolyRes(len(all_mods), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        elif mix_type == 'mmvae':
            mix_vae = MMVAEPolyRes(len(all_mods), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        elif mix_type == 'mvae':
            mix_vae = MVPolyRes(len(all_mods), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)
        elif mix_type == 'mvt':
            mix_vae = MVTPolyRes(len(all_mods), enc_channel_list, dec_channel_list, size_z, size_in, img_ch)

    optimizer = torch.optim.Adam(mix_vae.parameters(), lr=lr)
    mix_vae = mix_vae.to(device)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # print("test", flush=True)
    # mix_vae([torch.randn(100,3,28,28).to(device)]*10)
    # print("passed test", flush=True)
    print("Mods: ", all_mods, flush=True)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(pclf_path, map_location=device))
    poly_clf = poly_clf.to(device)
    poly_clf.eval()
    
    if vae_type == 'res':
        train_dataloader, val_dataloader, test_dataloader = get_train_test_dataloader_upd10_32x32(batch_size, mix_type=mix_type)
    # else:
    #     train_dataloader, test_dataloader, _ = get_train_test_dataloader_upd10(batch_size)
    
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

        training_loss = train_model(train_dataloader, mix_vae, mix_type, optimizer, all_mods, device, kl_cons)
        validation_loss = evaluate_model(val_dataloader, mix_vae, mix_type, all_mods, device, epoch, vae_type, True, kl_cons)
        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        # scheduler.step(validation_loss[0])

        # if epoch == 0:
        #     prev_loss = validation_loss[0]
        # if epoch > 0 and (validation_loss[0] < prev_loss):
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
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss[0],
            'val_loss': validation_loss[0],
            'beta': kl_cons,
            'size_z': size_z,
            }, "./models/" + mix_type + '/' + mix_type + "NEW" + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z) + all_mods + '_' + unq_name + 'fid_' + str(average_fid)[:4] + '_epoch_' + str(epoch + 1))
            print('Model saved', flush=True)
            print(" ")

    print("best average fid: ", best_average_fid, ' at epoch: ', best_epoch, flush=True)

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/' + mix_type + '/' + mix_type + "_vae_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z) + all_mods + '_' + unq_name)  
    
    shutil.rmtree(temp_dir_name)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--all-mods', type=str, default='0123456789',
                        help='all mods [default: "0123456789"]')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size for training [default: 512]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--cuda', type=int, default=3,
                        help='number of cuda_gpu [default: 3]')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta value for kl loss [default: 0.5]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--unq-name', type=str, default='_AN_',
                        help='identifier name for saving [default: "_AN_"]')
    parser.add_argument('--mix-type', type=str, default='mopoe',
                        help='mixture type [default: "mopoe"]')
    parser.add_argument('--vae-type', type=str, default='res',
                        help='vae type [default: "res"]')
    parser.add_argument('--pclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='poly classifier path [default: "./models/pm_clf/pm_clf_best"]')
    
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.unq_name, args.vae_type, args.mix_type, args.all_mods, args.cuda, args.pclf_path)