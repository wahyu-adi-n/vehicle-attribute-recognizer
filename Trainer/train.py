import torch
import argparse
import os
import torch.nn as nn
from utils.metrics import ClassificationMetrics
from comet_ml import Artifact, Experiment
from utils.utils import generate_model_config, read_cfg, get_optimizer, get_device, generate_hyperparameters, save_model, save_plots, SaveBestModel
from datasets.dataset import CarsDataModule
from tqdm.auto import tqdm
from models.models import create_model
from utils.logger import get_logger
import argparse
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    logger.log_metric("train_loss", epoch_loss)
    logger.log_metric("train_accuracy", epoch_acc)

    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))

    logger.log_metric("val_loss", epoch_loss)
    logger.log_metric("val_accuracy", epoch_acc)

    return epoch_loss, epoch_acc


def train(model, train_loader, valid_loader, cfg):
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    epochs = cfg['train']['num_epochs']
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate_one_epoch(
            model, valid_loader, criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, cfg)
        print('-'*50)
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, cfg)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, cfg)
    print('TRAINING COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Argument for train the model")
    parser.add_argument('-cfg', '--config', default="/content/drive/Shareddrives/skripsi/VAR/vehicle-attribute-recognizer/Trainer/configs/effnetb0.yaml",
                        type=str, help="Path to config yaml file")
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    hyperparameters = generate_hyperparameters(cfg)

    LOG = get_logger(cfg['model']['backbone'])

    LOG.info("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'],
                        project_name=cfg['logger']['project_name'],
                        workspace=cfg['logger']['workspace'])  # logger for track model in Comet ML
    artifact = Artifact("Cars Artifact", "Model")
    LOG.info("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    LOG.info(f"{str(device)} has choosen.")
    print(f"Computation device: {device}")
    print(f"Epoch: {cfg['train']['num_epochs']}")
    print(f"Learning Rate: {cfg['train']['lr']}")
    print(f"Batch Size: {cfg['train']['batch_size']}\n")

    # Load the model
    kwargs = dict(weights=cfg['model']['weights'],
                  output_class=cfg['model']['num_classes'], fine_tune=True)
    backbone = create_model(
        model_name=cfg['model']['backbone'],
        **kwargs).to(device=device)
    LOG.info(f"Backbone {cfg['model']['backbone']} succesfully loaded.")

    optimizer = get_optimizer(cfg, backbone)
    LOG.info(f"Optimizer has been defined.")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel()
                                 for p in backbone.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Loss function.
    criterion = nn.CrossEntropyLoss()
    LOG.info(f"Criterion has been defined")

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    CarsData = CarsDataModule(cfg)
    train_dl = CarsData.train_dataloader()
    val_dl = CarsData.val_dataloader()
    LOG.info(f"Dataset train and val loader successfully loaded.")

    logger.log_parameters(hyperparameters)
    LOG.info("Parameters has been Logged")

    generate_model_config(cfg)
    LOG.info("Model config has been generated")

    train(backbone, train_dl, val_dl, cfg)

    best_model_path = os.path.join(cfg['output_dir'], 'best_model.pth')
    final_model_path = os.path.join(cfg['output_dir'], 'final_model.pth')
    model_cfg_path = os.path.join(cfg['output_dir'], 'model-config.yaml')
    acc_fig = os.path.join(cfg['output_dir'], 'acc_figure.png')
    loss_fig = os.path.join(cfg['output_dir'], 'loss_figure.png')
    artifact.add(best_model_path)
    artifact.add(final_model_path)
    artifact.add(model_cfg_path)
    artifact.add(acc_fig)
    artifact.add(loss_fig)
    logger.log_artifact(artifact=artifact)
    logger.log_metrics()
