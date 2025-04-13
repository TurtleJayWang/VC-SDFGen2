import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import os
from math import sqrt

from trainer.BaseTrainer import BaseTrainer
from data.dataset import ShapeNetSDF, ShapeNetVoxel32
from trainer.DeepSDFTrainer import DeepSDFTrainer
from trainer.VCCNFTrainer import VCCNFTrainer

class FullDataset(Dataset):
    def __init__(self, shapenetsdf : ShapeNetSDF, shapenetvoxel32 : ShapeNetVoxel32):
        self.shapenetsdf = shapenetsdf
        self.shapenetvoxel32 = shapenetvoxel32
        
    def __len__(self):
        return len(self.shapenetsdf)
    
    def __getitem__(self, index):
        points, sdfs = self.shapenetsdf[index]
        voxel_data = self.shapenetvoxel32[index]
        return voxel_data, points, sdfs

class FullTrainer(BaseTrainer):
    def __init__(self, deepsdf_trainer : DeepSDFTrainer, vccnf_trainer : VCCNFTrainer, result_dir):
        self.latent_dim = deepsdf_trainer.latent_dim
        
        self.deepsdf_model = deepsdf_trainer.deepsdf_model
        self.vccnf_model = vccnf_trainer.vccnf_model
        
        self.shapenetsdf = deepsdf_trainer.deepsdf_dataset
        self.shapenetvoxel32 = vccnf_trainer.shapenetvoxel32_dataset
        
        self.deepsdf_result_dir = deepsdf_trainer.result_dir
        self.vccnf_result_dir = vccnf_trainer.result_dir
        
        self.model_infos = {
            "deepsdf" : {
                "model" : self.deepsdf_model,
                "init_lr" : 0.0
            },
            "vccnf" : {
                "model" : self.vccnf_model,
                "init_lr" : 0.0
            }
        }
        
        super().__init__(
            self.model_infos,
            2000, 24,
            result_dir,
            "loss_full.npz"
        )
        
    def set_dataset(self):
        self.full_dataset = FullDataset(self.shapenetsdf, self.shapenetvoxel32)
        self.training_dataset, _ = random_split(self.full_dataset, [0.8, 0.2], torch.Generator().manual_seed(42))
        self.training_loader = DataLoader(self.training_dataset, self.batch_size, True)
        
    def set_optimizer(self):
        self.deepsdf_optimizer = optim.Adam(self.deepsdf_model.parameters(), 0.0000675)
        self.vccnf_optimizer = optim.Adam(self.vccnf_model.parameters(), 0.000125)
        self.deepsdf_scheduler = optim.lr_scheduler.StepLR(self.deepsdf_optimizer, 1000, 0.5)
        self.vccnf_scheduler = optim.lr_scheduler.StepLR(self.vccnf_optimizer, 1000, 0.5)
        
        if self.get_latest_epoch() > 0:
            self.deepsdf_optimizer.load_state_dict(torch.load(os.path.join(self.result_dir, "deepsdf_optimizer.pth"), weights_only=True))
            self.vccnf_optimizer.load_state_dict(torch.load(os.path.join(self.result_dir, "vccnf_optimizer.pth"), weights_only=True))
            self.deepsdf_scheduler.load_state_dict(torch.load(os.path.join(self.result_dir, "deepsdf_scheduler.pth"), weights_only=True))
            self.vccnf_scheduler.load_state_dict(torch.load(os.path.join(self.result_dir, "vccnf_scheduler.pth"), weights_only=True))
        
    def save_optimizer(self):
        torch.save(self.deepsdf_optimizer.state_dict(), os.path.join(self.result_dir, "deepsdf_optimizer.pth"))
        torch.save(self.vccnf_optimizer.state_dict(), os.path.join(self.result_dir, "vccnf_optimizer.pth"))
        torch.save(self.deepsdf_scheduler.state_dict(), os.path.join(self.result_dir, "deepsdf_scheduler.pth"))
        torch.save(self.vccnf_scheduler.state_dict(), os.path.join(self.result_dir, "vccnf_scheduler.pth"))
        
    def epoch_train(self, epoch):
        epoch_loss = 0
        self.vccnf_model.train()
        self.deepsdf_model.train()
        for i, (voxels, points, sdfs) in enumerate(self.training_loader):
            voxels = voxels.to(self.device)
            points = points.to(self.device)
            sdfs = sdfs.to(self.device)
            
            batch_size = voxels.shape[0]
            
            mean = 0.0
            std = 1 / sqrt(self.latent_dim)
            gaussian_latents = torch.randn((batch_size, self.latent_dim)) * std + mean
            gaussian_latents = gaussian_latents.to(self.device)
            
            latent_codes = self.vccnf_model(voxels, gaussian_latents)
            sdf_preds = self.deepsdf_model(latent_codes, points)
            
            loss = F.l1_loss(sdf_preds, sdfs, reduction="sum")
            
            self.deepsdf_optimizer.zero_grad()
            self.vccnf_optimizer.zero_grad()
            
            loss.backward()
            
            self.deepsdf_optimizer.step()
            self.vccnf_optimizer.step()
            
            epoch_loss += loss.item()

        self.deepsdf_scheduler.step()
        self.vccnf_scheduler.step()
        
        return epoch_loss