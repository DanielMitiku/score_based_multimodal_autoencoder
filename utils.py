import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch.nn as nn


def save_model(model, path):
    torch.save(model, path)
    print('Model saved as ' + path, flush=True)

def save_model_dict(model, path):
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path, flush=True)

def get_saved_model(path):
    return torch.load(path)

def save_loss_plot(train_loss, val_loss, title, xlabel, ylabel, path):
    for ind in range(len(train_loss[0])):
        plt.figure()
        plt.plot([i for i in range(len(train_loss))], train_loss[:,ind])
        plt.plot([i for i in range(len(train_loss))], val_loss[:,ind])
        plt.title(title[ind])
        plt.xlabel(xlabel[ind])
        plt.ylabel(ylabel[ind])
        plt.legend(['Train', 'Val'])
        plt.savefig(path +str(ind) + '.png')

def save_fid_plot(fids, title, legend, path):
    plt.figure()
    for ind in range(len(fids[0])):
        plt.plot([i for i in range(len(fids))], fids[:,ind])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.legend(legend)
    plt.savefig(path + '.png')

def save_fid_plot_single(fids, title, path):
    plt.figure()
    plt.plot([i for i in range(len(fids))], fids)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.savefig(path + '.png')

def save_loss_plot_single(loss, title, path):
    plt.figure()
    plt.plot([i for i in range(len(loss))], loss)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(path + '.png')

def save_loss_plot_train_val(loss1, loss2, title, legend, path):
    plt.figure()
    plt.plot([i for i in range(len(loss1))], loss1)
    plt.plot([i for i in range(len(loss2))], loss2)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(path + '.png')


def save_batch_image(batch, path):
    for i in range(batch.shape[0]):
        save_image(batch[i], path + str(i) + '.png')
        
        
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def set_seed(seed=42, n_gpu=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)