import torch.nn as nn
import torchvision.models as models
import torch


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__(input_dim, output_dim, hidden_dims, num_layers = 3 ,drop_p=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.fcs = nn.ModuleList()
        for i in range(1, num_layers):
            if i == 0:
                self.fcs.append(nn.Linear(self.input_dim, self.hidden_dims))
            else:
                self.fcs.append(nn.Linear(self.hidden_dims, self.hidden_dims))

        self.fcs.append(nn.Linear(self.hidden_dims, self.output_dim))
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for fc in self.fcs:
            x = self.activation(fc(x))
        return x


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        # create image module as resnet18 using pytorch
        self.image_module = models.resnet18(pretrained=True)
        for param in self.image_module.parameters():
            param.requires_grad = False

        self.image_module = nn.Sequential(*list(self.image_module.children())[:-1])

        self.measured_speed_module = FullyConnectedNet(1, 128, 128, 3)
        self.speed_pred_module = FullyConnectedNet(512, 1, 128, 3)
        self.join_module = FullyConnectedNet(640, 512, 640, 1)
        #create 4 fully connected layers for control
        self.control_module = nn.ModuleList(FullyConnectedNet(512, 1, 128, 3) for _ in range(4))
         #FullyConnectedNet(512, 1, 128, 3)



    def forward(self, img, speed, command):
        p_i = self.image_module(img) #shape: (batch_size, 512, 1, 1)
        p_i = p_i.view(p_i.size(0), -1) #flatten for speed pred module #shape: (batch_size, 512)

        
        v = self.measured_speed_module(speed) #shape: (batch_size, 128)
        #concatenate v and p_i to shape (batch_size, 640)
        p_i = torch.cat((p_i, v), dim=1) #shape: (batch_size, 640)
        x = self.join_module(p_i) #shape: (batch_size, 512)
        v_p = self.speed_pred_module(p_i) #shape: (batch_size, 1)

        #create list of control outputs
        control_outputs = []
        for i in range(4):
            control_outputs.append(self.control_module[i](x))

        return v_p, control_outputs






