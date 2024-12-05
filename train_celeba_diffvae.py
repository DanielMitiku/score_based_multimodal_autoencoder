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
import math
from inspect import isfunction
from functools import partial
from einops import rearrange
from torch import nn, einsum

from h_vae_model import ConvCelebA, SigConvCelebA
from h_vae_model_copy import ResVAEN, ResAEN
# from unet_model import UnetVAE
from unet_openai import UNetModel
from utils import *
from celeba_hq_mask_dataset import CelebAHQMaskDS

    
# """Taken from NVAE implementation at https://github.com/NVlabs/NVAE/"""
# class CropCelebA64(object):
#     """ This class applies cropping for CelebA64. This is a simplified implementation of:
#     https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
#     """
#     def __call__(self, pic):
#         new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
#         return new_pic

#     def __repr__(self):
#         return self.__class__.__name__ + '()'

# def get_train_test_dataloader(batch_size):
#     train_transform = transforms.Compose([
#         CropCelebA64(),
#         transforms.Resize((64,64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
#     val_transform =  transforms.Compose([
#         CropCelebA64(),
#         transforms.Resize((64,64)),
#         transforms.ToTensor(),
#     ])

#     train_dataset = torchvision.datasets.CelebA(root='data', split='train', transform=train_transform, download=True)
#     val_dataset = torchvision.datasets.CelebA(root='data', split='valid', transform=val_transform, download=True)

#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return train_dataloader, val_dataloader

def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    # beta_end = 0.0005
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 1000
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def diff_losses(denoise_model, x_start, z_image, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(torch.cat([x_noisy, z_image],dim=1), t)

    return F.mse_loss(noise, predicted_noise)


def train(train_loader, sm_model, optimizer, image_vae, device):
    losses = 0
    sm_model.train()
    image_vae.eval()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = images.to(device)
        input_norm = (input * 2) - 1
        
        with torch.no_grad():
            # Get x_hat
            x_hat, _, _ = image_vae(input)
            # x_hat = image_vae(input)
            x_hat = (x_hat * 2) - 1
        
        with torch.enable_grad():
            t = torch.randint(0, timesteps, (input.shape[0],), device=device).long()
            loss = diff_losses(sm_model, input_norm, x_hat, t)
            # loss = diff_losses(sm_model, torch.cat([input_norm,x_hat],dim=1), t)

            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    losses /= len(train_loader)
    print("TRAINING TIME TAKEN: ", time.time() - start_time, flush=True)
    print('train loss: ', losses, flush=True)
    return losses

@torch.no_grad()
def evaluate(val_loader, sm_model, image_vae, epoch, unq_name, device):
    losses = 0
    sm_model.eval()
    image_vae.eval()
    diff_outs = []
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(val_loader):

        input = images.to(device)
        input_norm = (input * 2) - 1
        
        with torch.no_grad():
            # Get x_hat
            x_hat, _, _ = image_vae(input)
            # x_hat = image_vae(input)
            x_hat = (x_hat * 2) - 1
        
            t = torch.randint(0, timesteps, (input.shape[0],), device=device).long()
            loss = diff_losses(sm_model, input_norm, x_hat, t)
            # loss = diff_losses(sm_model, torch.cat([input_norm,x_hat],dim=1), t)

            losses += loss.item()

    sample_in = input[0].unsqueeze(0)
    vae_out, _, _ = image_vae(sample_in)
    # vae_out = image_vae(sample_in)
    diff_outs.extend([sample_in, vae_out])
    vae_out = (vae_out * 2) -1
    sample = torch.normal(mean=0, std=1, size=sample_in.shape, device=device)

    for t in range(timesteps-1, -1, -1):

        t_batch = torch.full((sample.shape[0],), t, device=device, dtype=torch.long)
        betas_t = extract(betas, t_batch, sample.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t_batch, sample.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_batch, sample.shape)
        
        model_mean_t = sqrt_recip_alphas_t * (
            # sample - betas_t * sm_model(sample, vae_out, t_batch) / sqrt_one_minus_alphas_cumprod_t
            sample - betas_t * sm_model(torch.cat([sample, vae_out], dim=1), t_batch) / sqrt_one_minus_alphas_cumprod_t
        )

        if t > 0:
            posterior_variance_t = extract(posterior_variance, t_batch, sample.shape)
            noise = torch.randn_like(sample)
            sample = model_mean_t + torch.sqrt(posterior_variance_t) * noise
        else:
            sample = model_mean_t
        
        if t == 0:
            diff_outs.append((sample + 1) / 2)
        

    diff_outs = torch.cat(diff_outs, dim=0)
    concat_tensor = torchvision.utils.make_grid(diff_outs)
    plt.imshow(concat_tensor.permute(1,2,0).detach().cpu().numpy())
    plt.title('diff')
    plt.axis("off")
    plt.savefig('./images/diff/diff_vae_' + str(epoch) + '__' + str(batch_idx) +  unq_name + '.png')

    losses /= len(val_loader)
    print('val loss: ', losses, flush=True)
    print("Validation TIME TAKEN: ", time.time() - start_time, flush=True)
    return losses

def run(epochs, batch_size, lr, size_z, vae_path, unq_name):
    res_size = 128
    print('vars: ', epochs, batch_size, lr, size_z, unq_name, res_size, flush=True)
    train_losses = []
    val_losses = []
    show = True

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:4")
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    # size_in = 64
    # img_ch = 3    
    # image_vae = SigConvCelebA(size_z)
    # image_vae.load_state_dict(torch.load(vae_path))
    # image_vae = image_vae.to(device)
    # image_vae.eval()
    # enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2), (512,1024,1024,2)]
    # dec_channel_list = [(1024,1024,512,2), (512,512,256,2), (256,256,128,2), (128,128,64,2)]
    #sm
    enc_channel_list = [(64,128,128,2), (128,256,256,2), (256,512,512,2)]
    dec_channel_list = [(512,512,256,2), (256,256,128,2), (128,128,64,2)]
    size_in = res_size
    img_ch = 3 

    image_vae = ResVAEN(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    image_vae.load_state_dict(torch.load(vae_path)['model_state_dict'])
    image_vae = image_vae.to(device)

    unetvae = UNetModel(in_channels=img_ch*2, model_channels=128, out_channels=img_ch, num_res_blocks=2, attention_resolutions=(16,), dropout=0.1, channel_mult=(1,2,2,3,4), num_heads=8)
    unetvae.load_state_dict(torch.load('./models/diff/diff_vae__diffvae_hq_1000_resvaeN_openai_')['model_state_dict'])
    optimizer = torch.optim.Adam(unetvae.parameters(), lr=lr)
    unetvae = unetvae.to(device)

    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, res_size)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)
        training_loss = train(train_dataloader, unetvae, optimizer, image_vae, device)
        validation_loss = evaluate(val_dataloader, unetvae, image_vae, epoch, unq_name, device)
        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)

        # if epoch == 0:
        #     prev_loss = validation_loss
        # if epoch > 0 and (validation_loss < prev_loss):
        torch.save({
        'epoch': epoch,
        'model_state_dict': unetvae.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': training_loss,
        'val_loss': validation_loss,
        }, "./models/diff/diff_vae_" + str(unq_name))
        print('Model saved', flush=True)
        # prev_loss = validation_loss

        if (epoch + 1) % 50 == 0:
            lr /= 5
            optimizer = torch.optim.Adam(unetvae.parameters(), lr=lr)

        
    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot_train_val(train_losses, val_losses, 'Loss', ['Train', 'Val'], './plots/diff/diff_vae_' + unq_name + ".png") 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta value for kl loss [default: 1]')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training [default: 64]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate [default: 0.0001]')
    parser.add_argument('--unq-name', type=str, default='_diffvae_hq_1000_resvaeN_openai_agn_',
                        help='identifier name for saving [default: "_diffvae_hq_1000_resvaeN_openai_agn_"]')
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_1.0_res_celebhq',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_1.0_res_celebhq"]')
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_ae__dsize_128_z_1024_1kae_sm_hq_',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_ae__dsize_128_z_1024_1kae_sm_hq_"]')
    # parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_ae__dsize_128_z_256_res_celebhq_ae_',
    #                     help='vae model path [default: "./models/celeba/celeb_hq_ae__dsize_128_z_256_res_celebhq_ae_"]')
    parser.add_argument('--image-path', type=str, default='./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__',
                        help='vae model path [default: "./models/celeba/celeb_hq_res_dsize_128_z_256_beta_0.1_smN_256__"]')


    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.image_path, args.unq_name)


