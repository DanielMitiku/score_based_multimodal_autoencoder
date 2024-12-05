from statistics import mean
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
import os
from PIL import Image

from celeba_hq_mask_dataset import CelebAHQMaskDS

from h_vae_model import ResCelebA
from h_vae_model_copy import ResVAEN

from utils import *
    

def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader

def vae_loss(x, x_hat, mu, logvar, model, kl_cons):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss / x.shape[0], kl_cons * kl_loss / x.shape[0]

def train_model(train_loader, model, optimizer, epoch, device, kl_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = images.to(device)

        out, mu, logvar = model(input)
        rec, kl = vae_loss(input, out, mu, logvar, model, kl_cons)
        losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()

        # print('out: ', out.mean().item(), 'mu: ', mu.mean().item(), 'logvar: ', logvar.mean().item(), 'rec loss: ', rec.item(), 'kl_loss1: ', kl.item(), 'kl_loss2: ', mean_inside_kls.item(), flush=True) 
        total_loss = rec + kl 
        losses['total'] += total_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
        optimizer.step()

        if epoch < 1 and batch_idx < 1:
            print('done one batch', flush=True)

    end_time = time.time()
    for loss in losses:
        losses[loss] /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return list(losses.values())

def evaluate_model(val_loader, model, device, epoch, unq_name, res_size, show=True, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, masks, target) in enumerate(val_loader):

            input = images.to(device)

            out, mu, logvar = model(input)
            rec, kl = vae_loss(input, out, mu, logvar, model, kl_cons)
            losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()

            total_loss = rec + kl
            losses['total'] += total_loss.item()

            if show and (np.random.uniform() < 0.3):
                # Plot random input and output
                sample_idx = torch.randint(input.shape[0], size=(1,)).item()
                sample_in = input[sample_idx].permute(1,2,0).to(device)
                sample_out = out[sample_idx].permute(1,2,0).to(device)

                concat_tensor = torch.cat([sample_in, sample_out], dim=1)
                concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('    Input   Output  ')
                plt.axis("off")
                plt.savefig('./images/celeba/celeb_hq_res_' + '_dsize_' + str(res_size) + str(kl_cons) + '_' + str(model.size_z) + '__' + str(epoch) + '__' + str(batch_idx) +  unq_name + '.png')

        # plot samples
        samples = model.sample(100, device)
        grid = torchvision.utils.make_grid(samples, nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig('./images/celeba/celeb_hq_res_' + '_dsize_' + str(res_size) + str(kl_cons) + '_' + str(model.size_z) + '__' + str(epoch) + unq_name + '.png')

        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        return list(losses.values())


def run(res_size, epochs, batch_size, lr, size_z, beta, unq_name):
    print('vars: ', res_size, epochs, batch_size, lr, size_z, beta, unq_name, flush=True)
    train_losses = []
    val_losses = []
    kl_cons = beta
    show = True

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:4")
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    # enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2), (512,1024,1024,2)]
    # dec_channel_list = [(1024,1024,512,2), (512,512,256,2), (256,256,128,2), (128,128,64,2)]
    #sm
    enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
    dec_channel_list = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]

    size_in = res_size
    img_ch = 3    
    celeba_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    optimizer = torch.optim.Adam(celeba_vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)
    celeba_vae = celeba_vae.to(device)

    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, res_size)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, celeba_vae, optimizer, epoch, device, kl_cons)
        validation_loss = evaluate_model(val_dataloader, celeba_vae, device, epoch, unq_name, res_size, show, kl_cons)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        # scheduler.step(validation_loss[0])

        if epoch == 0:
            prev_loss = validation_loss[0]
        if epoch > 0 and (validation_loss[0] < prev_loss):
            torch.save({
            'epoch': epoch,
            'model_state_dict': celeba_vae.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'size_z': size_z,
            }, "./models/celeba/celeb_hq_res" + "_dsize_" + str(res_size) + "_z_" + str(size_z) + "_beta_" + str(kl_cons) + str(unq_name))
            print('Model saved', flush=True)
            prev_loss = validation_loss[0]

        if (epoch + 1) % 100 == 0:
            lr /= 5
            optimizer = torch.optim.Adam(celeba_vae.parameters(), lr=lr)

        
    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/celeba/celeb_hq_res' + "_dsize_" + str(res_size) + "_beta_" + str(kl_cons) + "__" + str(size_z) +  unq_name + ".png")    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--res-size', type=int, default=128,
                        help='resolution size for the dataset [default: 128]')
    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta value for kl loss [default: 1]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    parser.add_argument('--unq-name', type=str, default='_res_celebhq',
                        help='identifier name for saving [default: "_res_celebhq"]')

    args = parser.parse_args()

    run(args.res_size, args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.unq_name)


