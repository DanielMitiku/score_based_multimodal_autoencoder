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
from h_vae_model import CelebAAttrNewBN
from sklearn.metrics import f1_score

from utils import *
    
def get_train_test_dataloader(batch_size, size):
    train_dataset = CelebAHQMaskDS(size=size, ds_type='train')
    val_dataset = CelebAHQMaskDS(size=size, ds_type='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader

def vae_loss(x, x_hat, mu, logvar, model, kl_cons):
    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    recon_loss = bce_logit_loss(x_hat, x)
    # mse = nn.MSELoss(reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss/ x_hat.shape[0], kl_cons*(kl_loss / x_hat.shape[0])

def train_model(train_loader, attr_visible, model, optimizer, epoch, device, kl_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, masks, target) in enumerate(train_loader):

        target = target.to(device)[:, attr_visible]

        out, mu, logvar = model(target.float())
        rec, kl = vae_loss(target.float(), out, mu, logvar, model, kl_cons)
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

def evaluate_model(val_loader, attr_visible, model, device, epoch, show=True, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        model.eval()
        correct, total = 0, 0
        f1_avg = 0
        start_time = time.time()

        for batch_idx, (images, masks, target) in enumerate(val_loader):

            target = target.to(device)[:, attr_visible]

            out, mu, logvar = model(target.float())
            rec, kl = vae_loss(target.float(), out, mu, logvar, model, kl_cons)
            losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()

            # sigmoid_outputs = torch.sigmoid(out).cpu()
            # predicted = np.round(sigmoid_outputs)
            sigmoid_outputs = torch.sigmoid(out).cpu()
            predicted = np.round(sigmoid_outputs) 
            total += target.shape[0] * target.shape[1]
            correct += (predicted == target.cpu()).sum().item()
            f1_avg += f1_score(target.cpu().numpy(), predicted, average='samples')

            total_loss = rec + kl
            losses['total'] += total_loss.item()


        accuracy = correct / total
        f1_avg /= len(val_loader)
        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        print("Validation accuracy and f1: ", accuracy, f1_avg, flush=True)
        return list(losses.values())


def run(epochs, batch_size, lr, size_z, beta):
    print('vars: ', epochs, batch_size, lr, size_z, beta, flush=True)
    attr_visible  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]

    train_losses = []
    val_losses = []
    kl_cons = beta
    show = True

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:4")

    celeba_attr = CelebAAttrNewBN(size_z)
    optimizer = torch.optim.Adam(celeba_attr.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10)
    celeba_attr = celeba_attr.to(device)

    train_dataloader, val_dataloader = get_train_test_dataloader(batch_size, 128)

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, attr_visible, celeba_attr, optimizer, epoch, device, kl_cons)
        validation_loss = evaluate_model(val_dataloader, attr_visible, celeba_attr, device, epoch, show, kl_cons)

        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        # scheduler.step(validation_loss[1])

        if epoch == 0:
            prev_loss = validation_loss[0]
        if epoch > 0 and (validation_loss[0] < prev_loss):
            torch.save({
            'epoch': epoch,
            'model_state_dict': celeba_attr.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss,
            'val_loss': validation_loss,
            'size_z': size_z,
            }, "./models/celeba_attr/celeba_attr_bn_hq_" + "_z_" + str(size_z) + "_beta_" + str(kl_cons))
            print('Model saved', flush=True)
            prev_loss = validation_loss[0]

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/celeba_attr/celeba_attr_bn_hq_' + "_beta_" + str(kl_cons) + "_lr_" + str(lr) + "__" + str(size_z) + ".png")    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=256,
                        help='size of z [default: 256]')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta value for kl loss [default: 1]')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size for training [default: 512]')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train [default: 300]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')

    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.beta)