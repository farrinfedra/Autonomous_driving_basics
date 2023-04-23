from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as transforms

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here
        self.data = []
        image_path = os.path.join(self.data_root, 'rgb')
        measurements_path = os.path.join(self.data_root, 'measurements')

        for file in os.listdir(image_path):

            img_path = os.path.join(image_path, file)
            json_path = os.path.join(measurements_path, file[:-3] + 'json')
            self.data.append((img_path, json_path))
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        img_path, json_path = self.data[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        

        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]),
                ])
        img = transform(img)
         #channel, height, width: 3, 224, 224 => 
                            #when input to model unsqueeze(0) to add batch dimension

        #convert to numpy array

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        #add measurements as dictionary 
        measurements = {}
        measurements['speed'] = torch.tensor(json_data['speed'], dtype=torch.float)
        measurements['throttle'] = torch.tensor(json_data['throttle'], dtype=torch.float)
        measurements['steer'] = torch.tensor(json_data['steer'], dtype=torch.float)
        measurements['brake'] = torch.tensor(json_data['brake'], dtype=torch.float)                                                      
        measurements['command'] = torch.tensor(json_data['command'], dtype=torch.long)
        measurements['tl_state'] = torch.tensor(json_data['tl_state'], dtype=torch.long)
        measurements['tl_dist'] = torch.tensor(json_data['tl_dist'], dtype=torch.float)
        measurements['lane_dist'] = torch.tensor(json_data['lane_dist'], dtype=torch.float)
        measurements['route_angle'] = torch.tensor(json_data['route_angle'], dtype=torch.float)

                                                
        return img, measurements
        