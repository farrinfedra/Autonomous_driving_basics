from torch.utils.data import Dataset
import os
import json
import numpy as np
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

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        img_path, json_path = self.data[index]
        img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])
        img = transform(img) #channel, height, width: 3, 224, 224 => 
                            #when input to model unsqueeze(0) to add batch dimension

        #convert to numpy array
        img = np.array(img)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            
        measurements = np.array([json_data['speed'], json_data['throttle'], json_data['brake'],
                        json_data['steer'], json_data['command']])
        # measurements = np.array([json_data['speed'], json_data['throttle'], json_data['brake'],
        #                          json_data['steer'], json_data['command'], json_data['route_dist'],
        #                          json_data['route_angle'], json_data['lane_dist'], json_data['lane_angle'],
        #                          json_data['tl_state'], json_data['tl_dist'], json_data['is_junction']])
        return img, measurements
