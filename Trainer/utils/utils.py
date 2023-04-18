from easydict import EasyDict as edict
from torch import optim
from torch import device
import os
import yaml
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, cfg):
        save_dir = cfg['output_dir'] + cfg['model']['backbone']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, os.path.join(save_dir, f'best_model_{cfg.model.backbone}.pth'))


def save_model(epochs, model, optimizer, criterion, cfg):
    save_dir = cfg['output_dir'] + cfg['model']['backbone']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(save_dir, f'final_model_{cfg.model.backbone}.pth'))


def save_plots(train_acc, valid_acc, train_loss, valid_loss, cfg):
    save_dir = cfg['output_dir'] + cfg['model']['backbone']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, linestyle='-', label='Train Accuracy')
    plt.plot(valid_acc, linestyle='-', label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy plot model {cfg.model.backbone}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'acc_figure_{cfg.model.backbone}.png'))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, linestyle='-', label='Train Loss')
    plt.plot(valid_loss, linestyle='-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss plot model {cfg.model.backbone}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_figure_{cfg.model.backbone}.png'))

def read_cfg(cfg_file):
    with open(cfg_file, 'r') as rf:
        cfg = edict(yaml.safe_load(rf))
        return cfg

def get_device(cfg):
    result_device = None
    if cfg['device'] == 'cpu':
        result_device = device("cpu")
    elif cfg['device'] == 'cuda:0':
        result_device = device("cuda:0")
    else:
        raise NotImplementedError
    return result_device

def get_optimizer(cfg, backbone):
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(backbone.parameters(), 
                              lr=cfg['train']['lr'], 
                              weight_decay=cfg['train']['weight_decay'])
                              
    elif cfg['train']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(backbone.parameters(), 
                              lr=cfg['train']['lr'],
                              weight_decay=cfg['train']['weight_decay'])

    elif cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(backbone.parameters(), 
                              lr=cfg['train']['lr'],
                              momentum=cfg['train']['momentum'],
                              weight_decay=cfg['train']['weight_decay'],
                              )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(backbone.parameters(), 
                              lr=cfg['train']['lr'],
                              weight_decay=cfg['train']['weight_decay'])
    else:
        raise NotImplementedError
    return optimizer

def generate_hyperparameters(train_cfg: dict):
    return {
        'model': train_cfg.model.backbone,
        'model_input': train_cfg.model.input_size[0],
        'output_class': train_cfg.model.num_classes,
        'batch_size': train_cfg.train.batch_size,
        'optimizer': train_cfg.train.optimizer,
        'learning_rate': train_cfg.train.lr,
        'weight_decay': train_cfg.train.weight_decay,
        'epoch': train_cfg.train.num_epochs,
    }

def generate_model_config(train_cfg: dict):
    model_config = {
        'model': train_cfg.model.backbone,
        'model_input': train_cfg.model.input_size[0],
        'output_class': train_cfg.model.num_classes,
        'model_file': f'best_model_{train_cfg.model.backbone}.pth'
    }
    save_dir = train_cfg['output_dir'] + train_cfg.model.backbone

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'model-config-{train_cfg.model.backbone}.yaml')
    with open(save_path, "w") as yaml_file:
        yaml.safe_dump(model_config, yaml_file, sort_keys=False,
                       explicit_start=True, default_flow_style=None)
