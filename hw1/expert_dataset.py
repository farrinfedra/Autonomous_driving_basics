from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from PIL import Image
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
        img = Image.open(img_path)

        transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]),
                ])
        img = transform(img) #channel, height, width: 3, 224, 224 => 
                            #when input to model unsqueeze(0) to add batch dimension

        #convert to numpy array
        img = np.array(img)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        #add measurements as dictionary 
        measurements = {}
        measurements['speed'] = torch.Tensor(json_data['speed']).type(torch.float)
        measurements['throttle'] = torch.Tensor(json_data['throttle']).type(torch.float)
        measurements['steer'] = torch.Tensor(json_data['steer']).type(torch.float)
        measurements['brake'] = torch.Tensor(json_data['brake']).type(torch.float)                                                      
        measurements['command'] = torch.Tensor(json_data['command']).type(torch.long)
        measurements['tl_state'] = torch.Tensor(json_data['tl_state']).type(torch.long)
        measurements['tl_dist'] = torch.Tensor(json_data['tl_dist']).type(torch.float)
        measurements['lane_dist'] = torch.Tensor(json_data['lane_dist']).type(torch.float)
        measurements['route_angle'] = torch.Tensor(json_data['route_angle']).type(torch.float)

                                                
        return img, measurements
        