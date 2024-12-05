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
from h_vae_model_copy import ResAEND, ResAEN, ResAE

from utils import *
    

def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader

# def ae_loss(x, x_hat):
#     mse = nn.MSELoss(reduction='sum')
#     recon_loss = mse(x_hat, x)
#     return recon_loss / x.shape[0]

def ae_reg_loss(x, x_hat, z):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    z_norm = 1 * torch.mean(torch.sum(z ** 2, dim=-1))
    return recon_loss / x.shape[0], z_norm

def train_model(train_loader, model, optimizer, epoch, device, kl_cons, noise_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = images.to(device)

        # out = model(input)
        # total_loss = ae_loss(input, out)

        z = model.encoder(input)
        out = model.decoder(z + noise_cons * torch.randn_like(z))
        
        recon, norm =  ae_reg_loss(input, out, z)
        total_loss = recon + kl_cons * norm

        losses['total'] += total_loss.item()
        losses['recs'] += recon.item()
        losses['kls'] += kl_cons * norm.item()
        
        optimizer.zero_grad()
        total_loss.backward()
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

            # out = model(input)
            # total_loss = ae_loss(input, out)

            z = model.encoder(input)
            out = model.decoder(z)

            recon, norm =  ae_reg_loss(input, out, z)
            total_loss = recon + kl_cons * norm

            losses['total'] += total_loss.item()
            losses['recs'] += recon.item()
            losses['kls'] += kl_cons * norm.item()

            if show and (np.random.uniform() < 0.2):
                # Plot random input and output
                sample_idx = torch.randint(input.shape[0], size=(1,)).item()
                sample_in = input[sample_idx].permute(1,2,0).to(device)
                sample_out = out[sample_idx].permute(1,2,0).to(device)

                concat_tensor = torch.cat([sample_in, sample_out], dim=1)
                concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('    Input   Output  ')
                plt.axis("off")
                plt.savefig('./images/celeba/celeb_hq_ae__' + '_dsize_' + str(res_size) + str(kl_cons) + '_' + str(model.size_z) + '__' + str(epoch) + '__' + str(batch_idx) +  unq_name + '.png')

        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        return list(losses.values())


def run(res_size, epochs, batch_size, lr, size_z, beta, noise_cons, unq_name):
    print('vars: ', res_size, epochs, batch_size, lr, size_z, beta, noise_cons, unq_name, flush=True)
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
    #sm2
    # enc_channel_list = [(128,256,256,4), (256,512,512,2)]
    # dec_channel_list = [(512,512,256,2), (256,256,128,4)]

    size_in = res_size
    img_ch = 3    
    celeba_vae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    optimizer = torch.optim.Adam(celeba_vae.parameters(), lr=lr)
    celeba_vae = celeba_vae.to(device)

    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, res_size)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, celeba_vae, optimizer, epoch, device, kl_cons, noise_cons)
        validation_loss = evaluate_model(val_dataloader, celeba_vae, device, epoch, unq_name, res_size, show, kl_cons)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)

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
            }, "./models/celeba/celeb_hq_ae_" + "_beta_" + str(kl_cons) + "_noisecons_" + str(noise_cons) + "_z_" + str(size_z) + str(unq_name))
            print('Model saved', flush=True)
            prev_loss = validation_loss[0]

        if (epoch + 1) % 40 == 0:
            lr /= 5
            optimizer = torch.optim.Adam(celeba_vae.parameters(), lr=lr)

    print(" ", flush=True)
    print("*************** best model loss *****************", flush=True)
    celeb_vae = ResAEN(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    model_name = "./models/celeba/celeb_hq_ae_" + "_beta_" + str(kl_cons) + "_noisecons_" + str(noise_cons) + "_z_" + str(size_z) + str(unq_name)
    celeb_vae.load_state_dict(torch.load(model_name)['model_state_dict'])
    celeb_vae = celeb_vae.to(device)
    evaluate_model(val_dataloader, celeb_vae, device, epoch, unq_name, res_size, show, kl_cons)

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/celeba/celeb_hq_ae_' + "_beta_" + str(kl_cons) + "_noisecons_" + str(noise_cons) + "__" + str(size_z) + str(unq_name) + ".png")    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--res-size', type=int, default=128,
                        help='resolution size for the dataset [default: 128]')
    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--beta', type=float, default=1e-5,
                        help='beta value for kl loss [default: 1e-5]')
    parser.add_argument('--noise-cons', type=float, default=0.01,
                        help='noise cons value [default: 0.01]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    parser.add_argument('--unq-name', type=str, default='_res_celebhq_ae_',
                        help='identifier name for saving [default: "_res_celebhq_ae_"]')

    args = parser.parse_args()

    run(args.res_size, args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.noise_cons, args.unq_name)


