import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import random
from math import sqrt

class BaseTrainer:
    def __init__(self, model_infos : dict, datasets, epochs, batch_size, result_dir, loss_file_name, model_save_frequency=100):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        self.model_infos = model_infos
        self.epochs = epochs
        self.batch_size = batch_size
        
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        self.model_file_names = list(self.model_infos.keys())
        self.models = []
        for model_name, model_info in self.model_infos.items():
            self.models.append(model_info["model"])

        for i, model in enumerate(self.models):
            self.models[i] = model.to(self.device)
            self.models[i].train()
        
        self.loss_file_name = loss_file_name
        self.result_dir = result_dir
        
        self.model_save_frequency = model_save_frequency
        
        self.load_datasets(datasets)
        
        model_lr = {}
        for model_name, model_info in self.model_infos.items():
            model = model_info["model"]
            lr = model_info["init_lr"]
            model_lr[model_name] = {
                "params" : model.parameters(),
                "lr" : lr
            }
        self.set_optimizer(model_lr)
            
    def __del__(self):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
            
    def __len__(self):
        return self.epochs - self.get_latest_epoch()
        
    def __iter__(self):
        self.pre_train()
        
        for e in range(self.start_epoch, self.epochs):
            epoch = e + 1
            epoch_loss = self.epoch_train(epoch)
            
            self.losses.append(epoch_loss)

            if epoch % self.model_save_frequency == 0:
                self.save_models(epoch)
                self.save_loss()
                self.save_optimizer()
                
            yield epoch, self.losses
        
    def epoch_train(self, epoch):
        """In epoch train, we will train the model for one epoch. All you need to do is manage the tensor dimensions and the data loader."""
        pass
    
    def load_datasets(self, datasets):
        pass
        
    def set_optimizer(self, model_lr : dict):
        pass
    
    def save_optimizer(self):
        pass
    
    def pre_train(self):
        self.load_loss()
    
        self.start_epoch = self.get_latest_epoch()
        if self.start_epoch > 0:
            self.load_models(self.start_epoch)
    
    def save_models(self, epoch):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(self.result_dir, f"{self.model_file_names[i]}_epoch_{epoch}.pth"))
            
    def load_models(self, epoch):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(self.result_dir, f"{self.model_file_names[i]}_epoch_{epoch}.pth"), weights_only=True))
    
    def load_loss(self):
        if os.path.exists(os.path.join(self.result_dir, self.loss_file_name) + ".npy"):
            self.losses = np.load(os.path.join(self.result_dir, self.loss_file_name) + ".npy").tolist()
            self.losses = self.losses[:self.get_latest_epoch()]
        else:
            self.losses = []
    
    def save_loss(self):
        np.save(os.path.join(self.result_dir, self.loss_file_name), np.array(self.losses))
    
    def get_latest_epoch(self):
        if os.path.exists(os.path.join(self.result_dir, self.loss_file_name) + ".npy"):
            losses = np.load(os.path.join(self.result_dir, self.loss_file_name) + ".npy").tolist()
            if losses:
                latest_epoch = len(losses) // self.model_save_frequency * self.model_save_frequency
                is_true_latest = True
                for model_file_name in self.model_file_names:
                    if not os.path.exists(os.path.join(self.result_dir, f"{model_file_name}_epoch_{latest_epoch}.pth")):
                        is_true_latest = False
                        break
                while not is_true_latest:
                    latest_epoch -= self.model_save_frequency
                    is_true_latest = True
                    for model_file_name in self.model_file_names:
                        if not os.path.exists(os.path.join(self.result_dir, f"{model_file_name}_epoch_{latest_epoch}.pth")):
                            is_true_latest = False
                            break
                return latest_epoch
            else:
                return 0
        else:
            return 0