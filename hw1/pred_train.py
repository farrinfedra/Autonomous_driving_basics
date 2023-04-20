import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import wandb
wandb.init(project="CVAD", name="affordances")
device = "cuda" if torch.cuda.is_available() else "cpu"
def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.L1Loss()
    model.eval()
    loss_avg = 0
    count = 0

    for i, (img, measurements) in enumerate(tqdm(dataloader, desc="Validation")):
        img = img.to(device)
        measurements = {k: v.to(device) for k, v in measurements.items()}
        lane_dist, route_angle, tl_dist, tl_state = model(img, measurements['command'])
        
        classification_loss = classification_criterion(tl_state, measurements['tl_state'])
        regression_loss = regression_criterion(tl_dist, measurements['tl_dist']) + regression_criterion(lane_dist, measurements['lane_dist']) +  regression_criterion(route_angle, measurements['route_angle'])
        loss = classification_loss + regression_loss
        loss_avg += loss.item()
        count += 1
        wandb.log({"val_loss": loss_avg / count})

    return loss_avg / count


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    loss_avg = 0
    count = 0
    for i, (img, measurements) in enumerate(tqdm(dataloader, desc="Training")):
        img = img.to(device)
        measurements = {k: v.to(device) for k, v in measurements.items()}

        optimizer.zero_grad()
        lane_dist, route_angle, tl_dist, tl_state = model(img, measurements['command'])
        
        classification_loss = classification_criterion(tl_state, measurements['tl_state'])
        regression_loss = regression_criterion(tl_dist, measurements['tl_dist']) + regression_criterion(lane_dist, measurements['lane_dist']) +  regression_criterion(route_angle, measurements['route_angle'])
        loss = classification_loss + regression_loss
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        count += 1
        wandb.log({"train_loss": loss_avg / count})
    return loss_avg / count


def plot_losses(train_loss, val_loss, epochs):
    """Visualize your plots and save them for your report."""
    # Your code here
    x = range(1, epochs+1)
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, val_loss, label="val loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"cilrs_loss_{epochs}.png")

    


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    root = "/userfiles/ssafadoust20/expert_data"
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")

    
    print("Using device: {}".format(device))

    model = AffordancePredictor().to(device)
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True,
                            )

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        print("Epoch {}/{}".format(i + 1, num_epochs))
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
        torch.save(model, f"pred_model_ep{i}.ckpt")
    plot_losses(train_losses, val_losses, num_epochs)


if __name__ == "__main__":
    main()
