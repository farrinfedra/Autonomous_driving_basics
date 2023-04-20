import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F

class TaskBlock(nn.Module):
    """Task block that takes images as input and outputs task-specific features"""
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                output_dim, 
                dropout=0.5,
                conditional=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.conditional = conditional

        if not self.conditional:
            self.layer = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        else:
            self.layer = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.output_dim)
                ) for _ in range(4) #number of commands
            ])

    def forward(self, features, command=None):
        if not self.conditional:
            return self.layer(features)
        else:
            return torch.stack([self.layer[command[i]](features[i]) for i in range(len(command))])


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self, input_dim=512, hidden_dim=64, dropout_p=0.5):
        super().__init__()
        #super(AffordancePredictor, self).__init__()
        self.image_module = models.resnet18(pretrained=True)
        for param in self.image_module.parameters():
            param.requires_grad = False

        self.image_module.eval()

        self.image_module = nn.Sequential(*list(self.image_module.children())[:-1])
        
        self.lane_dist_module = TaskBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            output_dim=1, 
            dropout=dropout_p,
            conditional=True,
        )

        self.route_angle_module = TaskBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout_p,
            conditional=True,
        )

        self.tl_dist_module = TaskBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout_p,
            conditional=False,
        )

        self.tl_state_module = TaskBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            dropout=dropout_p,
            conditional=False,
        )

    def forward(self, img, command=None):
        with torch.no_grad():
            p_i = self.image_module(img)
        p_i = p_i.view(p_i.size(0), -1)
        
        #process the conditional ones
        lane_dist = self.lane_dist_module(p_i, command)
        route_angle = self.route_angle_module(p_i, command)

        #process the non-conditional ones
        tl_dist = self.tl_dist_module(p_i)
        tl_state = self.tl_state_module(p_i)

        return lane_dist.squeeze(), route_angle.squeeze(), tl_dist.squeeze(), tl_state.squeeze()






