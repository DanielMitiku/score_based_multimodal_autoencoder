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
from h_vae_model_copy import ResVAE

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *

def get_train_test_dataloader_upd10_32x32(batch_size):
    paired_train_dataset, paired_val_dataset, paired_test_dataset = get_train_test_dataset_upd10_32x32()
    train_dataloader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

def vae_loss(x, x_hat, mu, logvar, model, kl_cons):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_hat, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss / x.shape[0], kl_cons*kl_loss / x.shape[0]

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

def check_file_len(path, amount):
    initial_count = 0
    for p in os.listdir(path):
        if os.path.isfile(os.path.join(path, p)):
            initial_count += 1
    if (initial_count != amount):
        print('file len error: ', path, dir, flush=True)
        return False
    return True

def calc_fid(model, device, input_path, gen_path):
    amount = 10_000
    samples = sample_pvae(model, amount, device, model.size_z)
    save_batch_image(samples, gen_path)

    assert check_file_len(input_path, amount)
    assert check_file_len(gen_path, amount)

    fid = calculate_fid_given_paths([input_path, gen_path], 256, device, 2048, 2)
    return fid

def calc_fid32(model, device, input_path, gen_path):
    amount = 10_000
    samples = model.sample(amount, device).view(-1,3,32,32)[:,:,2:30,2:30]
    save_batch_image(samples, gen_path)

    assert check_file_len(input_path, amount)
    assert check_file_len(gen_path, amount)

    fid = calculate_fid_given_paths([input_path, gen_path], 256, device, 2048, 2)
    return fid

def train_model(train_loader, model, optimizer, modality, device, kl_cons):
    losses = {}
    losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
    model.train()
    start_time = time.time()

    for batch_idx, (images, target) in enumerate(train_loader):

        input = images[modality].to(device)

        out, mu, logvar = model(input)
        rec, kl = vae_loss(input, out, mu, logvar, model, kl_cons)
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

def sample_pvae(model, amount, device, size_z):
    samples = torch.randn(amount, size_z).to(device)
    image_out = torch.clip(model.pm_decoder(samples),0,1)
    return image_out

def evaluate_model(val_loader, model, device, epoch, modality, predictor, vae_type, temp_dir, show=True, kl_cons=1):
    with torch.no_grad():
        losses = {}
        losses['total'], losses['recs'], losses['kls'] = 0, 0, 0
        accuracies = 0
        model.eval()
        start_time = time.time()

        for batch_idx, (images, target) in enumerate(val_loader):

            input = images[modality].to(device)
            target = target.to(device)

            out, mu, logvar = model(input)
            rec, kl = vae_loss(input, out, mu, logvar, model, kl_cons)
            losses['recs'], losses['kls'] = losses['recs'] + rec.item(), losses['kls'] + kl.item()

            total_loss = rec + kl
            losses['total'] += total_loss.item()

            accuracies = accuracies + calc_accuracy2([out], target, predictor).item()

            if show and (np.random.uniform() < 0.05):
                # Plot random input and output
                sample_idx = torch.randint(input.shape[0], size=(1,)).item()
                sample_in = input[sample_idx].permute(1,2,0).to(device)
                sample_out = out[sample_idx].permute(1,2,0).to(device)

                concat_tensor = torch.cat([sample_in, sample_out], dim=1)
                concat_tensor = torchvision.utils.make_grid(concat_tensor)
                plt.imshow(concat_tensor.detach().cpu().numpy())
                plt.title('    Input   Output  ')
                plt.axis("off")
                plt.savefig('./images/polyupd10_' + modality + '/polyupd10_' + vae_type + str(kl_cons) + '_' + str(model.size_z) + '__' + str(epoch) + '__' + str(batch_idx) + '.png')

        # plot samples 
        if vae_type == 'res':
            samples = model.sample(100, device)
        else:
            samples = sample_pvae(model, 100, device, model.size_z)
        grid = torchvision.utils.make_grid(samples, nrow=10)
        plt.title('Samples')
        plt.axis("off")
        plt.imshow(grid.detach().permute(1,2,0).cpu().numpy())
        plt.savefig('./images/polyupd10_' + modality + '/sample_polyupd10_' + vae_type + str(kl_cons) + '_' + str(model.size_z) + '__' + str(epoch) + '.png')

        if (epoch+1) % 50 == 0:
            if vae_type == 'res':
                fid = calc_fid32(model, device, './data/Upd10MMNIST/val/m' + modality[1] + '/', temp_dir + '/')
            else:    
                fid = calc_fid(model, device, './data/Upd10MMNIST/val/m' + modality[1] + '/', temp_dir + '/')
            print("FID: ", str(fid), " epoch=", str(epoch), flush=True)

        end_time = time.time()
        for loss in losses:
            losses[loss] /= len(val_loader)
        accuracies = accuracies / len(val_loader)
        print("VALIDATION TIME TAKEN: ", end_time - start_time, flush=True)
        print("Validation loss: ", losses, flush=True)
        print("Accuracies: ", accuracies, flush=True)
        return list(losses.values()), accuracies


def run(epochs, batch_size, lr, size_z, modality, beta, polyclf_path, vae_type, cuda_num):
    print('vars: ', vae_type, epochs, batch_size, lr, size_z, modality, beta, 'cuda: ', cuda_num, flush=True)
    train_losses = []
    val_losses = []
    kl_cons = beta
    show = True

    cuda = torch.cuda.is_available()
    print("GPU Available: ", cuda, flush=True)
    device = torch.device("cuda:" + str(cuda_num))
    print("device: ", torch.cuda.get_device_properties(device), flush=True)

    temp_dir_name = './samples/tempupd10' + modality + str(torch.randint(100000, size=(1,)).item())
    for p in ['./models/polyupd10_' + modality, './plots/polyupd10_' + modality, './images/polyupd10_' + modality, temp_dir_name]:
        if not os.path.exists(p):
            os.makedirs(p)


    poly_clf = PMCLF()
    poly_clf.load_state_dict(torch.load(polyclf_path))
    poly_clf = poly_clf.to(device)

    if vae_type == 'res':
        enc_channel_list = [(64,64,64,2), (64,128,128,2), (128,256,256,2)]
        dec_channel_list = [(256,128,128,2), (128,128,64,2), (64,64,64,2)]
        size_in = 32
        img_ch = 3    
        pmvae = ResVAE(enc_channel_list, dec_channel_list, size_in, size_z, img_ch)
    
    optimizer = torch.optim.Adam(pmvae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    pmvae = pmvae.to(device)

    if vae_type == 'res':
        train_dataloader, val_dataloader, test_dataloader = get_train_test_dataloader_upd10_32x32(batch_size)
    # else:    
    #     train_dataloader, val_dataloader, test_dataloader = get_train_test_dataloader_upd10(batch_size)
    acc = 0

    for epoch in range(epochs):
        print("Epoch: "+str(epoch + 1), flush=True)

        training_loss = train_model(train_dataloader, pmvae, optimizer, modality, device, kl_cons)
        validation_loss, acc = evaluate_model(val_dataloader, pmvae, device, epoch, modality, poly_clf, vae_type, temp_dir_name, show, kl_cons)
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
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss[0],
            'val_loss': validation_loss[0],
            'accuracy': acc,
            'beta': kl_cons,
            'size_z': size_z,
            }, "./models/polyupd10_" + modality + "/polyupd10_" + modality + "_" + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z))
            print('Model saved', flush=True)
            prev_loss = validation_loss[0]

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    save_loss_plot(train_losses, val_losses, \
        ['Total loss', 'Recon', 'KL'],
        ['Epoch']*3, 
        ['Total loss', 'Recon', 'KL'], './plots/polyupd10_' + modality + '/polyupd10_' + vae_type + "_beta_" + str(kl_cons) + "__" + str(size_z))    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--size-z', type=int, default=64,
                        help='size of z [default: 64]')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda num [default: 0]')
    parser.add_argument('--m', type=str, default='m0',
                        help='modality to train [default: m0]')
    parser.add_argument('--vae-type', type=str, default='res',
                        help='vae type to use [default: res]')
    parser.add_argument('--upd', type=str, default='',
                        help='updated polymnist dataset [default: ]')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta value for kl loss [default: 1]')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for training [default: 256]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--polyclf-path', type=str, default='./models/pm_clf/pm_clf_best',
                        help='multimodal model path [default: "./models/pm_clf/pm_clf_best"]')

    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.lr, args.size_z, args.m, args.beta, args.polyclf_path, args.vae_type, args.cuda)