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
from h_vae_model import CelebAAttrNewBNAE, CelebAAttrNewBNAE40
from sklearn.metrics import f1_score

from utils import *
    
def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader

def ae_loss(x, x_hat):
    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    recon_loss = bce_logit_loss(x_hat, x)
    # mse = nn.MSELoss(reduction='sum')
    return recon_loss/ x_hat.shape[0]

def ae_reg_loss(x, x_hat, z):
    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    recon_loss = bce_logit_loss(x_hat, x)
    z_norm = 1 * torch.mean(torch.sum(z ** 2, dim=-1))
    return recon_loss / x.shape[0], z_norm

def train_model(train_loader, attr_visible, model, optimizer, epoch, device, kl_cons, noise_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        input = target.to(device)[:, attr_visible]
        # input = target.to(device)

        z = model.encoder(input.float())
        out = model.decoder(z + noise_cons * torch.randn_like(z))
        
        recon, norm =  ae_reg_loss(input.float(), out, z)
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

def evaluate_model(val_loader, attr_visible, model, device, epoch, show=True, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        model.eval()
        correct, total = 0, 0
        f1_avg = 0
        start_time = time.time()

        for batch_idx, (images, masks, target) in enumerate(val_loader):

            input = target.to(device)[:, attr_visible]
            # input = target.to(device)

            z = model.encoder(input.float())
            out = model.decoder(z)
            
            recon, norm =  ae_reg_loss(input.float(), out, z)
            total_loss = recon + kl_cons * norm

            losses['total'] += total_loss.item()
            losses['recs'] += recon.item()
            losses['kls'] += kl_cons * norm.item()

            # sigmoid_outputs = torch.sigmoid(out).cpu()
            # predicted = np.round(sigmoid_outputs)
            sigmoid_outputs = torch.sigmoid(out).cpu()
            predicted = np.round(sigmoid_outputs) 
            total += input.shape[0] * input.shape[1]
            correct += (predicted == input.cpu()).sum().item()
            f1_avg += f1_score(input.cpu().numpy(), predicted, average='samples')

            losses['total'] += total_loss.item()


        accuracy = correct / total
        f1_avg /= len(val_loader)
        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        print("Validation accuracy and f1: ", accuracy, f1_avg, flush=True)
        return list(losses.values()), f1_avg


def run(epochs, batch_size, lr, size_z, beta, noise_cons, unq_name):
    print('vars: ', epochs, batch_size, lr, size_z, beta, noise_cons, unq_name, flush=True)
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]

    train_losses = []
    val_losses = []
    kl_cons = beta
    show = True

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:5")

    celeba_attr = CelebAAttrNewBNAE(size_z=size_z)
    optimizer = torch.optim.Adam(celeba_attr.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10)
    celeba_attr = celeba_attr.to(device)
    print('model loaded', flush=True)

    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, 128)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, attr_visible, celeba_attr, optimizer, epoch, device, kl_cons, noise_cons)
        validation_loss, f1 = evaluate_model(val_dataloader, attr_visible, celeba_attr, device, epoch, show, kl_cons)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        # scheduler.step(validation_loss[1])

        if epoch == 0:
            prev_f1 = f1
        if epoch > 0 and (f1 > prev_f1):
            torch.save({
            'epoch': epoch,
            'model_state_dict': celeba_attr.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'size_z': size_z,
            }, "./models/celeba_attr/celeba_attr_bn_hq_AEreg_" + "_z_" + str(size_z) + '_' + str(kl_cons) + unq_name)
            print('Model saved', flush=True)
            prev_f1 = f1

        if (epoch + 1) % 70 == 0:
            lr /= 5
            optimizer = torch.optim.Adam(celeba_attr.parameters(), lr=lr)

    print(" ", flush=True)
    print("*************** best model loss *****************", flush=True)
    celeba_attr = CelebAAttrNewBNAE(size_z=size_z)
    model_name = "./models/celeba_attr/celeba_attr_bn_hq_AEreg_" + "_z_" + str(size_z) + '_' + str(kl_cons) + unq_name
    celeba_attr.load_state_dict(torch.load(model_name)['model_state_dict'])
    celeba_attr = celeba_attr.to(device)
    evaluate_model(val_dataloader, attr_visible, celeba_attr, device, epoch, show, kl_cons)

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/celeba_attr/celeba_attr_bn_hq40_AEreg_' + "_lr_" + str(lr) + "__" + str(size_z) + '_' + str(kl_cons) + unq_name + ".png")    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--beta', type=float, default=1e-5,
                        help='beta value for kl loss [default: 1e-5]')
    parser.add_argument('--noise-cons', type=float, default=0.01,
                        help='noise cons value [default: 0.01]')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size for training [default: 512]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    parser.add_argument('--unq-name', type=str, default='_att_ae_',
                        help='identifier name for saving [default: "_att_ae_"]')

    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.beta, args.noise_cons, args.unq_name)


