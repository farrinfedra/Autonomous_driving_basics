import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import omegaconf


from expert_dataset import ExpertDataset
from models.cilrs import CILRS

from tqdm import tqdm
import wandb
import random
import pytorch_lightning as pl
from datetime import datetime
import numpy as np
global global_step
global_step = 0

def calculate_loss(cfg, v_p, throttle, brake, steering, gt_speed, gt_actions, criterion, split='train'):
        """Calculate loss"""
      
        speed_loss = cfg.speed_weight * criterion(v_p.squeeze(1), gt_speed)
        brake_loss = cfg.brake_weight * criterion(brake, gt_actions['brake'])
        steer_loss = cfg.steer_weight * criterion(steering, gt_actions['steer'])
        throttle_loss = criterion(throttle, gt_actions['throttle'])


        action_loss = brake_loss + steer_loss + throttle_loss
        total = speed_loss + action_loss
        
        if cfg.use_wandb:
            wandb.log({
                    'global': global_step,
                    f'{split}/loss': total.item(),
                    f'{split}/throttle_loss': throttle_loss.item(),
                    f'{split}/steer_loss': steer_loss.item(),
                    f'{split}/brake_loss': brake_loss.item(),
                    f'{split}/speed_loss': speed_loss.item(),
                    f'{split}/action_loss': action_loss.item()
                })
        return total

def validate(model, dataloader, criterion, config, device="cuda"):
    """Validate model performance on the validation dataset"""
    # Your code here
     #better in paper
    model.eval()

    #val_loss = []
    val_avg = 0
    count = 0

    with torch.no_grad():
        for img, measurements in tqdm((dataloader), total=len(dataloader), desc="Validating"):
            img = img.to(device)
            gt_speed = measurements['speed'].to(device)
            gt_actions = {'brake': measurements['brake'].to(device),
                      'throttle': measurements['throttle'].to(device),
                      'steer': measurements['steer'].to(device)}
            command = measurements['command'].to(device)

            #print(img.shape, gt_speed.shape, gt_actions.shape)
            v_p, throttle, brake, steering  = model(img, gt_speed, command)
            loss = calculate_loss(config.train, v_p, throttle, brake, steering, gt_speed, gt_actions, criterion, split='val')
            #detach loss
            val_avg += loss.item()
            count +=1        

    return val_avg/count
    



def train(model, dataloader, optimizer, criterion, config, device="cuda"):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step
    
    model.train()
    
    #train_loss = []
    loss_avg = 0
    count = 0
    

    for img, measurements in tqdm((dataloader), total=len(dataloader), desc="Training"):
        # Your code here
        img = img.to(device)
        gt_speed = measurements['speed'].to(device)
        gt_actions = {'brake': measurements['brake'].to(device),
                      'throttle': measurements['throttle'].to(device),
                      'steer': measurements['steer'].to(device)}
        command = measurements['command'].to(device)

        optimizer.zero_grad()
        v_p, throttle, brake, steering  = model(img, gt_speed, command)
        loss = calculate_loss(config.train, v_p, throttle, brake, steering, gt_speed, gt_actions, criterion)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        count +=1
        global_step += 1

    return loss_avg/count

        

def plot_losses(train_loss, val_loss, epochs, save_path):
    """Visualize your plots and save them for your report."""
    x = range(1, epochs+1)
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, val_loss, label="val loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.savefig(os.path.join(save_path, f"cilrs_loss_{epochs}.png"))

def plot_losses_serperate(train_loss, val_loss, save_path):
    """Visualize your plots and save them for your report."""
    # Your code here

    plt.figure(figsize=(7, 5))
    plt.grid()
    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1')
    plt.savefig(os.path.join(save_path, f'CIRLS_noDrop_30_train_loss_sep.png'))
    plt.figure(figsize=(7, 5))
    plt.grid()
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1')
    plt.savefig(os.path.join(save_path, f'CIRLS_noDrop_30_val_loss_sep.png'))



def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    random.seed(datetime.now())
    pl.seed_everything(2000)
    
    config = omegaconf.OmegaConf.load("config.yaml")
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", config.wandb.name)
    os.makedirs(save_path, exist_ok=True)
    weights_path = os.path.join(save_path, "weights")
    os.makedirs(weights_path, exist_ok=True)
    plots_path = os.path.join(save_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    #write the copy of the config file into a json and save it in save_path
    omegaconf.OmegaConf.save(config, os.path.join(save_path, "config.yaml"))

    if config.train.use_wandb:
        wandb.init(project=config.wandb.project, name=config.wandb.name)
        wandb.config.update(config)

    root = config.train.data_path
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))
    
    

    model = CILRS().to(device)
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = config.train.num_epochs
    batch_size = config.train.batch_size
    
    # #take only 1/10 of the data
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset), 70))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 25))

    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=config.train.num_workers, 
                                pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=config.train.num_workers, 
                                pin_memory=True)

    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config.train.lr, 
                                     weight_decay=config.train.weight_decay)

    criterion = torch.nn.L1Loss().to(device)
    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        print("Epoch: {}".format(i))
        train_lss = train(model, train_loader, optimizer, criterion, config, device)
        val_lss = validate(model, val_loader, criterion, config, device)

        train_losses.append(train_lss)
        val_losses.append(val_lss)
        torch.save(model.state_dict(), os.path.join(weights_path, f"cilrs_model_ep{i+1}.ckpt"))

        if config.train.use_wandb:
            wandb.log({
                    'epoch': i,
                    'train/loss_epoch': train_lss,
                    'val/loss_epoch': val_lss
                })
    plot_losses(train_losses, val_losses, num_epochs, plots_path)
    plot_losses_serperate(train_losses, val_losses, save_path)
    torch.save(model, os.path.join(weights_path, f"cilrs_model_full_{num_epochs}.ckpt"))

    if config.train.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
