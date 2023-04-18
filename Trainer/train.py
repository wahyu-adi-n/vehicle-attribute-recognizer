from comet_ml import Artifact, Experiment
from utils.utils import generate_model_config, read_cfg, get_optimizer, \
      get_device, generate_hyperparameters, save_model, save_plots, SaveBestModel
from datasets.dataset import CarsDataModule
from tqdm.auto import tqdm
from models.models import create_model
from utils.logger import get_logger
from utils.metrics import ClassificationMetrics
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def train_one_epoch(model, 
                    train_loader,
                    optimizer, 
                    criterion, 
                    device,
                    epoch):
    
    print('Training process...')
    train_running_loss = 0.0
    train_running_correct = 0.0

    # set the model to train mode initially
    model.train()
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

        # get the inputs and assign them to cuda
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass + backward + optimize
        outputs = model(image)

        # Calculate the loss.
        loss = criterion(outputs, labels)
        
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)

        # Calculate the loss/acc later
        train_running_loss += loss.item()
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation.
        loss.backward()

        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / len(train_loader)
    epoch_acc = 100.0 * (train_running_correct / len(train_loader.dataset))
    
    # Log metric for train loss and accuracy
    logger.log_metric("train_loss", epoch_loss, epoch=epoch)
    logger.log_metric("train_accuracy", epoch_acc, epoch=epoch)

    return epoch_loss, epoch_acc


def validate_one_epoch(model, 
                      val_loader, 
                      criterion, 
                      device,
                      epoch):
    
    print('Validation process...')

    valid_running_loss = 0.0
    valid_running_correct = 0.0

    accuracy = precision = recall = f1 = 0.0
    eval_metrics = ClassificationMetrics()

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)
                        
            # Calculate the loss.
            loss = criterion(outputs, labels)

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)

            valid_running_loss += loss.item()
            valid_running_correct += (preds == labels).sum().item()

            classification_metrics = eval_metrics(labels.cpu(), preds.cpu())
            
            accuracy += classification_metrics['accuracy']
            precision += classification_metrics['precision']
            recall += classification_metrics['recall']
            f1 += classification_metrics['f1_score']
        
        total_val_dataset = len(val_loader)

        epoch_accuracy = accuracy / total_val_dataset
        epoch_precision = precision / total_val_dataset
        epoch_recall = recall / total_val_dataset
        epoch_f1_score = f1 / total_val_dataset

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / len(val_loader)
    epoch_acc = 100.0 * (valid_running_correct / len(val_loader.dataset))
    
    # Log metric for val loss and accuracy
    logger.log_metric("val_loss", epoch_loss, epoch=epoch)
    logger.log_metric("val_accuracy", epoch_acc, epoch=epoch)
    logger.log_metric("accuracy", epoch_accuracy, epoch=epoch)
    logger.log_metric("precision", epoch_precision, epoch=epoch)
    logger.log_metric("recall", epoch_recall, epoch=epoch)
    logger.log_metric("f1_score", epoch_f1_score, epoch=epoch)
    
    return epoch_loss, epoch_acc, epoch_accuracy, \
           epoch_precision, epoch_recall, epoch_f1_score

def train(model, 
          train_loader, 
          val_loader, 
          optimizer,  
          criterion, 
          device, 
          cfg):

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    epochs = cfg['train']['num_epochs'] 

    for epoch in range(1, epochs+1):
        print(f"[INFO]: Epoch {epoch} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)

        # switch the model to eval mode to evaluate on test data
        valid_epoch_loss, valid_epoch_acc,_,_,_,_ = validate_one_epoch(
            model, val_loader, criterion, device, epoch)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"loss: {train_epoch_loss:.3f}, accuracy: {train_epoch_acc:.3f}")
        print(f"val loss: {valid_epoch_loss:.3f}, val accuracy: {valid_epoch_acc:.3f}")

        # save the best model till now if we have the least loss in the current epoch
        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, cfg)
        print('-'*50)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, cfg)

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, cfg)
    
    print('TRAINING COMPLETE')

    return model, train_loss, train_acc, valid_loss, valid_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', 
                        default="../Trainer/configs/effnetb1.yaml",
                        type=str, help="Path to config yaml file")
                        
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    hyperparameters = generate_hyperparameters(cfg)

    LOG = get_logger(cfg['model']['backbone'])

    LOG.info("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'],
                        project_name=cfg['logger']['project_name'],
                        workspace=cfg['logger']['workspace']) 

    artifact = Artifact("Cars Artifact", "Model")
    LOG.info("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    LOG.info(f"{str(device)} has choosen.")

    print(f"\nComputation device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Epoch: {cfg['train']['num_epochs']}")
    print(f"Learning Rate: {cfg['train']['lr']}")
    print(f"Optimizer: {cfg['train']['optimizer']}")
    print(f"Weight Decay: {cfg['train']['weight_decay']}")
    print(f"Batch Size: {cfg['train']['batch_size']}\n")

    # Load the model
    kwargs = dict(weights=cfg['model']['weights'],
                  output_class=cfg['model']['num_classes'], 
                  fine_tune=True)

    backbone = create_model(model_name=cfg['model']['backbone'],
                            **kwargs).to(device=device)
        
    LOG.info(f"Backbone {cfg['model']['backbone']} succesfully loaded.")
    print(backbone)

    optimizer = get_optimizer(cfg, backbone)
    LOG.info(f"Optimizer has been defined.")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Loss function.
    criterion = nn.CrossEntropyLoss()
    LOG.info(f"Criterion has been defined")

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    CarsData = CarsDataModule(cfg)
    train_dl = CarsData.train_dataloader()
    val_dl = CarsData.val_dataloader()
    test_dl = CarsData.test_dataloader()
    LOG.info(f"Dataset train, val, and test data loader successfully loaded.")

    logger.log_parameters(hyperparameters)
    LOG.info("Parameters has been Logged")

    generate_model_config(cfg)
    LOG.info("Model config has been generated")

    train(model = backbone, 
          train_loader = train_dl, 
          val_loader = val_dl,
          optimizer = optimizer, 
          criterion = criterion, 
          device = device, 
          cfg = cfg)
          
    save_dir = cfg['output_dir'] + cfg.model.backbone

    best_model_path = os.path.join(save_dir, f'best_model_{cfg.model.backbone}.pth')
    final_model_path = os.path.join(save_dir, f'final_model_{cfg.model.backbone}.pth')
    model_cfg_path = os.path.join(save_dir, f'model-config-{cfg.model.backbone}.yaml')
    acc_fig = os.path.join(save_dir, f'acc_figure_{cfg.model.backbone}.png')
    loss_fig = os.path.join(save_dir, f'loss_figure_{cfg.model.backbone}.png')
    
    artifact.add(best_model_path)
    artifact.add(final_model_path)
    artifact.add(model_cfg_path)
    artifact.add(acc_fig)
    artifact.add(loss_fig)

    logger.log_artifact(artifact=artifact)
    
    logger.end()