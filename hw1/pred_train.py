import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import omegaconf
import torch.nn.functional as F

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor

from tqdm import tqdm
import wandb
import random
import pytorch_lightning as pl
from datetime import datetime

global global_step
global_step = 0

def calculate_loss(cfg, gt_measure, pred_measure, criterion, class_criterion, split='train'):
        """Calculate loss"""
        #print("gt_measure['tl_state'][:, 1]: ", gt_measure['tl_state'].float())
        # print("pred_measure['tl_state'].float(): ", F.softmax(pred_measure['tl_state'][:, 1]))
        # print((gt_measure['tl_state'].dtype))
        # print((pred_measure['tl_state'].dtype))

        class_loss_tl_state = class_criterion( pred_measure['tl_state'], gt_measure['tl_state'])
        loss_tl_dist = criterion(gt_measure['tl_dist'], pred_measure['tl_dist'])
        loss_lane_dist = criterion(gt_measure['lane_dist'], pred_measure['lane_dist'])
        loss_route_angle = criterion(gt_measure['route_angle'], pred_measure['route_angle'])

        total = class_loss_tl_state + loss_tl_dist + loss_lane_dist + loss_route_angle

        if cfg.use_wandb:
            wandb.log({
                    'global': global_step,
                    f'{split}/loss': total.item(),
                    f'{split}/class_loss_tl_state': class_loss_tl_state.item(),
                    f'{split}/loss_tl_dist': loss_tl_dist.item(),
                    f'{split}/loss_lane_dist': loss_lane_dist.item(),
                    f'{split}/loss_route_angle': loss_route_angle.item(),
                })
        return total

def validate(model, dataloader, criterion, classification_criterion, config, device="cuda"):
    """Validate model performance on the validation dataset"""
    # Your code here
    model.eval()
    val_avg = 0
    count = 0

    with torch.no_grad():
    
        for img, measurements in tqdm((dataloader), total=len(dataloader), desc="Validating"):
            # Your code here
            img = img.to(device)
            gt_measurements = {k: v.to(device) for k, v in measurements.items()}

            lane_dist, route_angle, tl_dist, tl_state = model(img, gt_measurements['command'])

            pred_measurements = {'lane_dist': lane_dist, 
                            'route_angle': route_angle, 
                            'tl_dist': tl_dist, 
                            'tl_state': tl_state}
            
            loss = calculate_loss(config.train, 
                                    gt_measurements, 
                                    pred_measurements, 
                                    criterion, 
                                    classification_criterion)

            val_avg += loss.item()
            count +=1

    return val_avg/count
    



def train(model, dataloader, optimizer, criterion, classification_criterion, config, device="cuda"):
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
        gt_measurements = {k: v.to(device) for k, v in measurements.items()}

        optimizer.zero_grad()
        lane_dist, route_angle, tl_dist, tl_state = model(img, gt_measurements['command'])

        pred_measurements = {'lane_dist': lane_dist, 
                        'route_angle': route_angle, 
                        'tl_dist': tl_dist, 
                        'tl_state': tl_state}
        
        loss = calculate_loss(config.train, 
                                gt_measurements, 
                                pred_measurements, 
                                criterion, 
                                classification_criterion)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        count +=1
        global_step += 1

    return loss_avg/count

        

def plot_losses(train_loss, val_loss, epochs, save_path):
    """Visualize your plots and save them for your report."""
    # plot loss such that x-axis is the number of iterations and y-axis is the loss
    # Your code here
    x = range(1, epochs+1)
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, val_loss, label="val loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(os.path.join(save_path, f"affordance_loss_{epochs}.png"))





def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    random.seed(datetime.now())
    pl.seed_everything(2000)
    
    config = omegaconf.OmegaConf.load("affordance_config.yaml")
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
    
    

    model = AffordancePredictor().to(device)
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
    classification_criterion = torch.nn.CrossEntropyLoss().to(device)

    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        print("Epoch: {}".format(i))
        train_lss = train(model, train_loader, optimizer, criterion, classification_criterion, config, device)
        val_lss = validate(model, val_loader, criterion, classification_criterion, config, device)

        train_losses.append(train_lss)
        val_losses.append(val_lss)
        torch.save(model.state_dict(), os.path.join(weights_path, f"affordance_model_ep{i+1}.ckpt"))

        if config.train.use_wandb:
            wandb.log({
                    'epoch': i,
                    'train/loss_epoch': train_lss,
                    'val/loss_epoch': val_lss
                })
    plot_losses(train_losses, val_losses, num_epochs, plots_path)
    torch.save(model, os.path.join(weights_path, f"affordance_model_full_{num_epochs}.ckpt"))

    if config.train.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
