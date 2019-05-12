import os
import torch
import json
from torchvision import transforms
from model import UNet
from torch.optim import Adam, SGD, lr_scheduler
from torch.nn import CrossEntropyLoss, BCELoss

class Config:
    def __init__(self, config_path):
        
        self.config_path = config_path
        with open(config_path, 'r') as f:
            obj = json.load(f)
        self.log = obj['logs']
        self.img_size = (obj['image']['width'], obj['image']['height'])
        self.augflag = obj['image']['augmentation_flag']
        self.aug = obj['image']['augmentations']
        self.model = obj['training']['model']

        self.transform = self.compose_transforms(obj['image']['augmentations'])
        
        self.lr = obj['training']['lr']
        self.lr_decay = obj['training']['lr']
        
        self.ann_settings = obj['segmentation_mask']
        self.dtype = obj['segmentation_mask']['type']

        self.dataset_folders = obj['dataset_folders']
        
        self.training_params = obj['training']

    def compose_transforms(self, aug_list):
        
        tr = []
        for aug in aug_list:
            if 'args' in aug:
                tr.append(getattr(transforms, aug['name'])(**aug['args']))
            else:
                tr.append(getattr(transforms, aug['name']))

        return transforms.RandomChoice(tr)
    
    def store_train_config(self, train_config, num_classes):
        
        self.net_class = self.get_model(self.model, num_classes)
        self.lr = train_config['lr']
        self.lr_decay = train_config['lr_decay']
        self.optimizer = train_config['optimizer']
        self.decay_wait = train_config['decay_wait']
        self.decay_type = train_config['decay_type']

        if train_config['loss'] == 'cross_entropy':
            if num_classes == 2:
                self.loss = BCELoss()
            else:
                self.loss = CrossEntropyLoss()

    def get_model(self, mname, nclasses):     
        if mname == 'unet':
            return UNet(num_classes = nclasses)
    
    def set_num_classes(self, n):
        # add one class for background
        self.store_train_config(self.training_params, n+1)

    def get_optimizer(self, params):
        
        if self.optimizer == 'adam':
            self.optim = Adam(params, lr=self.lr)
        
        elif self.optimizer == 'sgd':
            
            self.optim = SGD(params, lr=self.lr, weight_decay=self.lr_decay)
            if self.decay_type == 'loss_plateau':
                self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True, patience=self.decay_wait)
        
        return self.optim
    
    def save_model(self, path, loss, epoch, module=False):
        
        if not module:
            torch.save(self.net_class.state_dict(), path)
        else:
            torch.save(self.net_class.module.state_dict(), path)

        with open(self.config_path, 'r') as f:
            obj = json.load(f)
        
        obj['logs']['load_from'] = os.path.abspath(path)

        if loss < obj['logs']['best_loss']:
            obj['logs']['best_epoch'] = epoch 
        
        obj['logs']['best_loss'] = min(obj['logs']['best_loss'], loss)
        obj['logs']['start_epoch'] = epoch+1

        with open(self.config_path, 'w') as f:
            json.dump(obj, f, indent=4)

    def load_model(self, path, module=False, location=None):
        
        if location == 'cpu':
            f = torch.load(path, map_location='cpu')
        else:
            f = torch.load(path)
        
        from collections import OrderedDict
        
        g = OrderedDict()
        for key in f.keys():
            g[key.replace('module.', '')] = f[key]
        f = g

        if not module:
            self.net_class.load_state_dict(f)
        else:
            self.net_class.module.load_state_dict(f)

if __name__=='__main__':
    
    #config = Config('train_config.json')
    config = Config('train_config2.json')
