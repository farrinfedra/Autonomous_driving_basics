import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from expert_dataset import ExpertDataset
from models.cilrs import CILRS

from tqdm import tqdm
import wandb

wandb.init(project="CVAD", name="Part1-no-dropout")

def validate(model, dataloader, device="cuda"):
    """Validate model performance on the validation dataset"""
    # Your code here
    criterion = torch.nn.L1Loss() #better in paper
    model.eval()

    #val_loss = []
    val_avg = 0
    with torch.no_grad():
        for i, (img, measurements) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating"):
            # Your code here
            img = img.to(device)
            gt_speed = measurements['speed'].to(device)
            gt_actions = measurements['actions'].to(device)
            command = measurements['command'].to(device)


            #print(img.shape, gt_speed.shape, gt_actions.shape)
            v_p, actions = model(img, gt_speed, command)
            loss = criterion(actions, gt_actions) + criterion(v_p.squeeze(1), gt_speed)
            #detach loss
            val_avg += loss.detach().item()
            
            wandb.log({"Validation Loss": val_avg/len(dataloader)})

    return val_avg/len(dataloader)
    


def train(model, dataloader, device="cuda"):
    """Train model on the training dataset for one epoch"""
    # Your code here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = torch.nn.L1Loss() #better in paper
    model.train()

    #train_loss = []
    loss_avg = 0
    for i, (img, measurements) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        # Your code here
        img = img.to(device)
        gt_speed = measurements['speed'].to(device)
        gt_actions = measurements['actions'].to(device)
        command = measurements['command'].to(device)

        optimizer.zero_grad()
        v_p, actions = model(img, gt_speed, command)
        #print(v_p.squeeze(1).shape, gt_speed.shape)
        #print(criterion(actions, gt_actions), criterion(actions, gt_actions).shape)
        loss = criterion(actions, gt_actions) + criterion(v_p.squeeze(1), gt_speed)
        loss.backward()
        optimizer.step()
        #detach loss
        loss_avg += loss.detach().item()

        wandb.log({"Training Loss": loss_avg/len(dataloader)})
    return loss_avg/len(dataloader)

        

def plot_losses(train_loss, val_loss, epochs):
    """Visualize your plots and save them for your report."""
    # plot loss such that x-axis is the number of iterations and y-axis is the loss
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))
    model = CILRS().to(device)
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"
    
    # #take only 1/10 of the data
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset), 70))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 25))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        print("Epoch: {}".format(i))
        train_lss = train(model, train_loader, device)
        val_lss = validate(model, val_loader, device)

        train_losses.append(train_lss)
        val_losses.append(val_lss)
        torch.save(model.state_dict(), f"cilrs_model_ep{i+1}.ckpt")

    plot_losses(train_losses, val_losses, num_epochs)
    torch.save(model, f"cilrs_model_full_{num_epochs}.ckpt")

if __name__ == "__main__":
    main()
    wandb.finish()
