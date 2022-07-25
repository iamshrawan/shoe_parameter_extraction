from torch.utils.data import Dataset
from torchvision import transforms
import torch

from pathlib import Path
from PIL import Image
import json
import os
import random
import glob
import numpy as np


class SVShoeDataset(Dataset):
    def __init__(self, data_dict, resize=(224,224)):
        self.shape_keys = data_dict['shape_keys']
        self.image_fns = data_dict['shoe_images']
        self.transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, idx):
        
        #Load Image
        render_img = Image.open(self.image_fns[idx]).convert('RGB')
        
        #Load json and get shape key tensor
        with open(self.shape_keys[idx], 'r') as j:
            shape_key_js = json.load(j)
        shape_key = torch.Tensor(shape_key_js['shapeKeyValues'])
        
        #Transform image
        
        render_img = self.transforms(render_img)
        
        return render_img, shape_key
    
    def __len__(self):
        return len(self.image_fns)
        
    
class SVShoeEdgeDataset(Dataset):
    def __init__(self, data_dict, resize=(224,224)):
        self.shape_keys = data_dict['shape_keys']
        self.image_fns = data_dict['shoe_images']
        self.transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,
                                 std=0.5)
        ])
        
    def __getitem__(self, idx):
        
        #Load Image
        render_img = Image.open(self.image_fns[idx])
        
        #Load json and get shape key tensor
        with open(self.shape_keys[idx], 'r') as j:
            shape_key_js = json.load(j)
        shape_key = torch.Tensor(shape_key_js['shapeKeyValues'])
        
        #Transform image
        
        render_img = self.transforms(render_img)
        
        return render_img, shape_key
    
    def __len__(self):
        return len(self.image_fns)
    

class Rotate(torch.nn.Module):

    def __init__(self, angle, fill):
        self.angle = angle
        self.fill = fill

    def __call__(self, img):
        return transforms.functional.rotate(img, angle=self.angle, fill=self.fill)
        
    def __repr__(self):
        return self.__class__.__name__+'()'

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, test_mode=False, resize=(224,224),\
                 num_angles=9, num_views=3, shuffle=True, single_view=False):
       
        self.root_dir = root_dir
        self.test_mode = test_mode
        self.num_angles = num_angles
        self.num_views = num_views
        self.single_view = single_view
        
        self.folders = sorted([f for f in os.listdir(self.root_dir) if not f.startswith('.')], key=int)
        temp = Image.open(glob.glob(os.path.join(self.root_dir, self.folders[0], '*.png'))[0])
        self.min_dim = np.min(temp.size)
        del temp

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.folders)/num_angles))
            folders_new = []
            for i in range(len(rand_idx)):
                folders_new.extend(self.folders[rand_idx[i]*num_angles:(rand_idx[i]+1)*num_angles])
            self.folders = folders_new


        self.transform = transforms.Compose([
            #transforms.CenterCrop(self.min_dim),
            transforms.Resize(resize),
            #transforms.RandomChoice([
             #   transforms.RandomHorizontalFlip(),
              #  transforms.RandomVerticalFlip()
            #]),
            #transforms.RandomRotation(degrees=10, fill=255, interpolation=transforms.InterpolationMode.BICUBIC ),
            transforms.ToTensor(),
            #transforms.RandomApply(torch.nn.ModuleList([Rotate(angle=90, fill=255)]),p=0.05),
            transforms.Normalize(mean=0.5,
                                 std=0.5)
        ])




    def __len__(self):
        return int(len(self.folders)/self.num_angles)


    def __getitem__(self, idx):
        # Retrieve nine folders from index idx
        folders = self.folders[idx*self.num_angles:(idx+1)*self.num_angles]
        
        #Load Shape keys from one of the folders
        json_file = glob.glob(os.path.join(self.root_dir, folders[0], '*.json'))[0]
        with open(json_file, 'r') as j:
            js = json.load(j)
        shape_keys = js['renderScenes'][0]['props'][0]['morphTargets']
        shape_keys_vec = torch.Tensor([target['value'] for target in shape_keys])
        shape_keys_vec = (shape_keys_vec - 0.5)/0.5
        
        # Sample one variation for each view
        imgs = []
        if self.single_view:
            view_k = folders[2*self.num_views:3*self.num_views]
            #a_view = random.sample(view_k, 1)[0]
            image_files = glob.glob(os.path.join(self.root_dir, view_k[0], '*.png'))[0]
            #image_file = random.sample(image_files, 1)[0]
            im = Image.open(image_files).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        elif self.num_views == 2:
            view_top = folders[0*3:1*3]
            view_front = folders[2*3:3*3]
            a_view_top = random.sample(view_top, 1)[0]
            a_view_front = random.sample(view_front, 1)[0]
            image_file_top = glob.glob(os.path.join(self.root_dir, a_view_top, '*.png'))[0]
            image_file_front = glob.glob(os.path.join(self.root_dir, a_view_front, '*.png'))[0]
            im_top = Image.open(image_file_top).convert('RGB')
            im_front = Image.open(image_file_front).convert('RGB')
            if self.transform:
                im_top = self.transform(im_top)
                im_front = self.transform(im_front)
            imgs.append(im_top)
            imgs.append(im_front)
        else:
            for i in range(self.num_views):
                view_k = folders[i*self.num_views:(i+1)*self.num_views]
                a_view = random.sample(view_k, 1)[0]
                image_files = glob.glob(os.path.join(self.root_dir, a_view, '*.png'))[0]
                #image_file = random.sample(image_files, 1)[0]
                im = Image.open(image_files).convert('RGB')
                if self.transform:
                    im = self.transform(im)
                imgs.append(im)

        return torch.stack(imgs), shape_keys_vec, idx
        

def get_dataset(mode, n_channel, split_path):
    if mode == 'train':
        with open(os.path.join(split_path, 'training.json'), 'r') as j:
            train_dict = json.load(j)
        dataset = SVShoeDataset(train_dict) if n_channel == 3 else SVShoeEdgeDataset(train_dict)
        
    elif mode == 'dev' or mode == 'validation':
        with open(os.path.join(split_path, 'dev.json'), 'r') as j:
            dev_dict = json.load(j)
        dataset = SVShoeDataset(dev_dict) if n_channel == 3 else SVShoeEdgeDataset(dev_dict)
    elif mode == 'test':
        with open(os.path.join(split_path, 'test.json'), 'r') as j:
            test_dict = json.load(j)
        dataset = SVShoeDataset(test_dict) if n_channel == 3 else SVShoeEdgeDataset(test_dict)
    
    return dataset

        