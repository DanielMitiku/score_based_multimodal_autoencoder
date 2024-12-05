#!/bin/bash

# Train individual VAEs

python train_celeba_res_hq.py --beta=0.1 1> cel_im_vae.out 2> cel_im_vae_error.out
python train_celeba_mask_hq.py --beta=1 1> cel_mask_vae.out 2> cel_mask_vae_error.out
python train_celeba_attr_hq.py --beta=0.1 1> cel_att_vae.out 2> cel_att_vae_error.out

# Train the score model (pass individual vae model paths using the correct path)

nohup python train_lat_celebhq_unet_cont2.py --beta0=0.1 --beta1=20 --N=1000 --ll-weighting=0 --noise-obs=1 --pc=1 --cuda=1 --vae-type="VAE" --sde-type="VPSDE" --size-z1=256 --size-z2=256 1> cel_sde.out 2> cel_sde_errr.out &

# Train individual AEs (pass individual ae model paths using the correct path)

python train_celeba_hq_ae.py --beta=1e-4 --noise-cons=0.001 1> cel_im_ae.out 2> cel_im_ae_error.out
python train_celeba_mask_hq_ae.py --beta=1e-5 --noise-cons=0.001 1> cel_mask_ae.out 2> cel_mask_ae_error.out
python train_celeba_attr_hq_ae.py --beta=1e-4 --noise-cons=0.1 1> cel_att_ae.out 2> cel_att_ae_error.out

# Train the score model

nohup python train_lat_celebhq_unet_cont2.py --beta0=0.1 --beta1=20 --N=1000 --ll-weighting=0 --noise-obs=1 --pc=1 --cuda=1 --vae-type="AE" --sde-type="VPSDE" --size-z1=256 --size-z2=256 1> celAE_sde.out 2> celAE_sde_errr.out &
