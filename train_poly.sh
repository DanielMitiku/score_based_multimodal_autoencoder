#!/bin/bash

# Train individual VAEs

python train_polyupd.py --m="m0"  --beta=0.5 1> poly_m0.out 2> poly_m0_error.out
python train_polyupd.py --m="m1"  --beta=0.5 1> poly_m1.out 2> poly_m1_error.out
python train_polyupd.py --m="m2"  --beta=0.5 1> poly_m2.out 2> poly_m2_error.out
python train_polyupd.py --m="m3"  --beta=0.5 1> poly_m3.out 2> poly_m3_error.out
python train_polyupd.py --m="m4"  --beta=0.5 1> poly_m4.out 2> poly_m4_error.out
python train_polyupd.py --m="m5"  --beta=0.5 1> poly_m5.out 2> poly_m5_error.out
python train_polyupd.py --m="m6"  --beta=0.5 1> poly_m6.out 2> poly_m6_error.out
python train_polyupd.py --m="m7"  --beta=0.5 1> poly_m7.out 2> poly_m7_error.out
python train_polyupd.py --m="m8"  --beta=0.5 1> poly_m8.out 2> poly_m8_error.out
python train_polyupd.py --m="m9"  --beta=0.5 1> poly_m9.out 2> poly_m9_error.out

# Train score model (pass individual vae model paths using the correct path)
nohup python train_poly_unet_cont.py --allmods="0123456789" --vae-type="VAE" --epochs=500 --pc=1 --beta0=1 --beta1=5 --noise-obs=1 --cuda=0 1> poly_sde.out 2> poly_sde_err.out &

# Train individual AEs

python train_polyupdAE.py --m="m0"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m0.out 2> polyAE_m0_error.out
python train_polyupdAE.py --m="m1"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m1.out 2> polyAE_m1_error.out
python train_polyupdAE.py --m="m2"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m2.out 2> polyAE_m2_error.out
python train_polyupdAE.py --m="m3"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m3.out 2> polyAE_m3_error.out
python train_polyupdAE.py --m="m4"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m4.out 2> polyAE_m4_error.out
python train_polyupdAE.py --m="m5"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m5.out 2> polyAE_m5_error.out
python train_polyupdAE.py --m="m6"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m6.out 2> polyAE_m6_error.out
python train_polyupdAE.py --m="m7"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m7.out 2> polyAE_m7_error.out
python train_polyupdAE.py --m="m8"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m8.out 2> polyAE_m8_error.out
python train_polyupdAE.py --m="m9"  --beta=1e-5 --noise-cons=0.01 1> polyAE_m9.out 2> polyAE_m9_error.out

# Train score model (pass individual ae model paths using the correct path)
nohup python train_poly_unet_cont.py --allmods="0123456789" --vae-type="AE" --epochs=500 --pc=1 --beta0=1 --beta1=5 --noise-obs=1 --cuda=0 1> polyAE_sde.out 2> polyAE_sde_err.out &
