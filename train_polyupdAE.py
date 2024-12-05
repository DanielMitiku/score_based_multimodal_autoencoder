from tempfile import tempdir
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np
import os

from polymnist_dataset import get_train_test_dataset_upd10_32x32
from polymnist_model import PMCLF
from h_vae_model_copy import ResAE

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_32x32()
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

def ae_loss(x, x_hat):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    return recon_loss / x.shape[0]

def ae_reg_loss(x, x_hat, z):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    z_norm = 1 * torch.mean(torch.sum(z ** 2, dim=-1))
    return recon_loss / x.shape[0], z_norm

def calc_accuracy2(image_out, label_in, predictor):
    predictor.eval()
    accuracies = np.zeros((len(image_out),))

    for i in range(len(image_out)):
        # Calculate image accuracy
        predicted_out = predictor(image_out[i].reshape(-1,3,28,28) if image_out[i].shape[-1] == 28 else image_out[i].view(-1,3,32,32)[:,:,2:30,2:30])
        predicted_out = torch.argmax(predicted_out, 1)
        img_acc = torch.sum(predicted_out == label_in).item()
        img_acc = img_acc / image_out[i].shape[0]
        accuracies[i] = img_acc

    return accuracies


def train_model(train_loader, model, optimizer, modality, device, kl_cons, noise_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):
        input = images[modality].to(device)

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

    end_time = time.time()
    for loss in losses:
        losses[loss] /= len(train_loader)
    print("TRAINING TIME TAKEN: ", end_time - start_time, flush=True)
    print("Training loss: ", losses, flush=True)
    return list(losses.values())

def sample_pvae(model, amount, device, size_z):
    samples = torch.randn(amount, size_z).to(device)
    image_out = torch.clip(model.pm_decoder(samples),0,1)
    return image_out

def evaluate_model(val_loader, model, device, epoch, modality, predictor, vae_type, show=True, kl_cons=1, noise_cons=0.1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        accuracies = 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, target) in enumerate(val_loader):

            input = images[modality].to(device)
            target = target.to(device)

            z = model.encoder(input)
            out = model.decoder(z)

            recon, norm =  ae_reg_loss(input, out, z)
            total_loss = recon + kl_cons * norm

            losses['total'] += total_loss.item()
            losses['recs'] += recon.item()
            losses['kls'] += kl_cons * norm.item()

            accuracies = accuracies + calc_accuracy2([out], target, predictor).item()

            if show and (np.random.uniform() < 0.1):
                # Plot random input and output
                sample_idx = torch.randint(input.shape[0], size=(1,)).item()
                sample_in = input[sample_idx].permute(1,2,0).to(device)
                sample_out = out[sample_idx].permute(1,2,0).to(device)

                concat_tensor = torch.cat([sample_in, sample_out], dim=1)
                concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('    Input   Output  ')
                plt.axis("off")
                plt.savefig('./images/polyupd10_' + modality + '/polyNEWAE_' + modality + str(model.size_z) + '_' + str(noise_cons) + '_' + str(kl_cons) + '__' + str(epoch) + '__' + str(batch_idx) + '.png')

        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        accuracies = accuracies / len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        print("Accuracies: ", accuracies, flush=True)
        return list(losses.values()), accuracies


def run(epochs, batch_size, lr, size_z, modality, beta, noise_cons, polyclf_path, vae_type):
    print('vars: ', vae_type, epochs, batch_size, lr, size_z, modality, beta, noise_cons, flush=True)
    train_losses = []
    val_losses = []
    kl_cons = beta
    show = True

    device = torch.device("cuda:4")
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    for p in ['./models/polyupd10_' + modality, './plots/polyupd10_' + modality, './images/polyupd10_' + modality]:
        if not os.path.exists(p):
            os.makedirs(p)

    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(polyclf_path))
    poly_clf = poly_clf.to(device)

    
    enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
    dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
    size_in = 32
    img_ch = 3    
    pmvae = ResAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    
    optimizer = torch.optim.Adam(pmvae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    pmvae = pmvae.to(device)

    train_dataloader, val_dataloader, test_dataloader = get_train_test_dataloader_upd10_32x32(batch_size)
    acc = 0

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, pmvae, optimizer, modality, device, kl_cons, noise_cons)
        validation_loss, acc = evaluate_model(val_dataloader, pmvae, device, epoch, modality, poly_clf, vae_type, show, kl_cons, noise_cons)
        scheduler.step(validation_loss[0])
        print(' ', flush=True)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        
        if epoch == 0:
            prev_loss = validation_loss[0]
        if epoch > 0 and (validation_loss[0] < prev_loss):
            torch.save({
            'epoch': epoch,
            'model_state_dict': pmvae.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'accuracy': acc,
            'beta': kl_cons,
            'size_z': size_z,
            }, "./models/polyupd10_" + modality + "/polyNEWAE_" + modality + "_" + str(size_z) + '_' + str(noise_cons) + '_' + str(kl_cons))
            print('Model saved', flush=True)
            prev_loss = validation_loss[0]

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/polyupd10_' + modality + '/polyNEWAE_' + str(size_z) + '_' + str(noise_cons) + '_' + str(kl_cons))
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--m', type=str, default='m0',
                        help='modality to train [default: m0]')
    parser.add_argument('--vae-type', type=str, default='AE',
                        help='vae type to use [default: AE]')
    parser.add_argument('--upd', type=str, default='',
                        help='updated polymnist dataset [default: ]')
    parser.add_argument('--beta', type=float, default=1e-7,
                        help='beta value for kl loss [default: 1e-7]')
    parser.add_argument('--noise-cons', type=float, default=0.01,
                        help='noise cons value [default: 0.01]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--polyclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='multimodal model path [default: "./models/pm_clf/pm_clf_best"]')

    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.m, args.beta, args.noise_cons, args.polyclf_path, args.vae_type)


