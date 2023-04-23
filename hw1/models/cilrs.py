import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, drop_p=0.0):
        super(FullyConnectedNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.drop_p = drop_p
        self.fcs = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.Dropout(self.drop_p),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.output_dim)
        )
    
    def forward(self, x):

        return self.fcs(x)


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        # create image module as resnet18 using pytorch
        self.image_module = models.resnet18(pretrained=True)
        # for param in self.image_module.parameters():
        #     param.requires_grad = False

        # self.image_module.eval()

        self.image_module = nn.Sequential(*list(self.image_module.children())[:-1])

        self.measured_speed_module = FullyConnectedNet(1, 128, 128)
        self.speed_pred_module = FullyConnectedNet(512, 1, 256)
        self.join_module = nn.Linear(640, 512)
        #create 4 fully connected layers for control
        self.control_module = nn.ModuleList([FullyConnectedNet(512, 3, 256) for _ in range(4)])
         #FullyConnectedNet(512, 1, 128, 3)



    def forward(self, img, speed, command):

        with torch.no_grad():
            p_i = self.image_module(img) #shape: (batch_size, 512, 1, 1)

        p_i = p_i.view(p_i.size(0), -1) #flatten for speed pred module #shape: (batch_size, 512)
        bs = img.size(0)

        v = self.measured_speed_module(speed.unsqueeze(1)) #shape: (batch_size, 128)

        #concatenate v and p_i to shape (batch_size, 640)
        joined = torch.cat([p_i, v], dim=1) #shape: (batch_size, 640)
        joined = self.join_module(joined) #shape: (batch_size, 512)
        v_p = self.speed_pred_module(p_i) #shape: (batch_size, 1)
        
        #action = torch.stack([self.control_module[command[i]](joined[i]) for i in range(len(command))])
        action = torch.zeros((bs, 3)).type(torch.FloatTensor).to(img.device)
            
        for i in range(bs):
            action[i] = self.control_module[command[i]](joined[i])
        #print("action shape: ", action.shape)
        # throttle, brake, steering = torch.sigmoid(action[:, 0])\
        #                             , torch.sigmoid(action[:, 1]),\
        #                             torch.tanh(action[:, 2])

        throttle = torch.sigmoid(action[:, 0])
        steering = torch.tanh(action[:, 1])
        brake = torch.sigmoid(action[:, 2])

        #action = {'throttle': throttle, 'brake': brake, 'steer': steering}

        return v_p, throttle, brake, steering






