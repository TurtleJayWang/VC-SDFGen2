from model.VCCNF import VCCNF
from data.dataset import ShapeNetVoxel32
from trainer.BaseTrainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import random
from math import sqrt
from torch.optim.lr_scheduler import StepLR

class VCCNFTrainer(BaseTrainer):
    def __init__(self, vccnf_model_info, dataset, results_dir, epochs, batch_size, embeddings):
        self.epochs = epochs
        self.batch_size = batch_size
        
        super().__init__(
            vccnf_model_info, [dataset], 
            epochs, batch_size,
            results_dir,
            "losses_vccnf_train",
            100
        )

        self.embeddings = embeddings
        self.vccnf_model = vccnf_model_info["vccnf"]["model"]
        self.embeddings = self.embeddings.to(self.device)
        self.vccnf_model = self.vccnf_model.to(self.device)
        
        self.loss_fn = nn.MSELoss()
        
    def load_datasets(self, datasets):
        self.shapenetvoxel64_dataset_splits = random_split(
            datasets[0], 
            [0.8, 0.2], 
            generator=torch.Generator().manual_seed(42)
        )
        self.shapenetvoxel64_loader_training = DataLoader(
            self.shapenetvoxel64_dataset_splits[0],
            batch_size=self.batch_size,
            shuffle=True
        )
        
    def set_optimizer(self, model_lr):
        self.optimizer = torch.optim.Adam(list(model_lr.values()), lr=1e-3)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        
        if self.get_latest_epoch() > 0:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.result_dir, "optimizer_vccnf.pth"), weights_only=True))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.result_dir, "scheduler_vccnf.pth"), weights_only=True))
        
    def save_optimizer(self):
        torch.save(self.optimizer, os.path.join(self.result_dir, "optimizer_vccnf.pth"))
        torch.save(self.scheduler, os.path.join(self.result_dir, "scheduler_vccnf.pth"))
        
    def epoch_train(self, epoch):
        epoch_loss = 0
        self.vccnf_model.train()
        for i, (voxel_data, indices) in enumerate(self.shapenetvoxel64_loader_training):
            indices = indices.to(self.device)
            latent_codes = self.embeddings(indices)
            latent_codes = latent_codes.to(self.device)
            
            gaussian_latents = torch.randn_like(latent_codes)
            gaussian_latents = gaussian_latents.to(self.device)
            
            voxel_data = voxel_data.to(self.device)
            
            latent_pred = self.vccnf_model(voxel_data, gaussian_latents)
            
            loss = self.loss_fn(latent_pred, latent_codes)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() 
        self.scheduler.step()
        self.losses.append(epoch_loss)