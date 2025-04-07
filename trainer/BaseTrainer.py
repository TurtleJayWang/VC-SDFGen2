import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import random
from math import sqrt

class BaseTrainer:
    def __init__(self, models, datasets, epochs, batch_size, result_dir, loss_file_name, model_file_names, model_save_frequency=100):
        self.models = models
        self.epochs = epochs
        self.batch_size = batch_size
        
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        self.load_datasets(datasets)
        
        models_parameters = []
        for model in self.models:
            models_parameters += list(model.parameters())
        self.set_optimizer(models_parameters)
        
        self.model_file_names = model_file_names
        self.loss_file_name = loss_file_name
        self.result_dir = result_dir
        
        self.model_save_frequency = model_save_frequency
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
    def __iter__(self):
        self.pre_train()
        
        for e in range(self.start_epoch, self.epochs):
            epoch = e + 1
            self.epoch_train(epoch)
            
            self.losses.append(self.loss)

            if e % self.model_save_frequency == 0:
                self.save_models(epoch)
                self.save_loss()
                
            yield e, self.losses
        
    def epoch_train(self, epoch):
        """In epoch train, we will train the model for one epoch. All you need to do is manage the tensor dimensions and the data loader."""
        pass
    
    def load_datasets(self, datasets):
        pass
        
    def set_optimizer(self, parameters):
        pass
    
    def pre_train(self):
        self.load_loss()
        
        for i, model in enumerate(self.models):
            self.models[i] = model.to(self.device)
            self.models[i].train()
    
        self.start_epoch = self.get_latest_epoch()
        if self.start_epoch > 0:
            self.load_models(self.start_epoch)
    
    def save_models(self, epoch):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(self.result_dir, f"{self.model_file_names[i]}_epoch_{epoch}.pth"))
            
    def load_models(self, epoch):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(self.result_dir, f"{self.model_file_names[i]}_epoch_{epoch}.pth")))
    
    def load_loss(self):
        if os.path.exists(os.path.join(self.result_dir, self.loss_file_name)):
            losses = np.load(os.path.join(self.result_dir, self.loss_file_name)).tolist()
        else:
            self.losses = []
    
    def save_loss(self):
        np.save(os.path.join(self.result_dir, self.loss_file_name), np.array(self.losses))
    
    def get_latest_epoch(self):
        if os.path.exists(os.path.join(self.result_dir, self.loss_file_name)):
            losses = np.load(os.path.join(self.result_dir, self.loss_file_name)).tolist()
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