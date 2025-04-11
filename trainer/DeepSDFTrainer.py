import sys
sys.path.append("..")

from model.DeepSDF import DeepSDF
from trainer.BaseTrainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from math import sqrt
import os

class DeepSDFTrainer(BaseTrainer):
    def __init__(self, deepsdf_model_infos, deepsdf_dataset, epochs, batch_size, results_dir, model_save_frequency=100):
        self.deepsdf_model : DeepSDF = deepsdf_model_infos["deepsdf"]["model"]
        
        self.latent_dim = self.deepsdf_model.latent_dim
        self.embeddings = nn.Embedding(len(deepsdf_dataset), self.latent_dim, max_norm=1.0)
        torch.nn.init.normal_(self.embeddings.weight.data, 0, 1 / sqrt(self.latent_dim))
        
        deepsdf_model_infos["latent_codes"] = {
            "model" : self.embeddings,
            "init_lr" : 1e-3
        }
        
        super().__init__(
            deepsdf_model_infos, [deepsdf_dataset], 
            epochs, batch_size,
            results_dir,
            "losses_deepsdf_train", 
            model_save_frequency
        )
    
    def load_datasets(self, datasets):
        self.deepsdf_dataset = datasets[0]
        self.deepsdf_train_dataloader = DataLoader(self.deepsdf_dataset, batch_size=self.batch_size, shuffle=True)
        
    def set_optimizer(self, model_lr : dict):    
        self.optimizer = torch.optim.Adam(list(model_lr.values()))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.scheduler.last_epoch = self.get_latest_epoch() - 1
        
        start_epoch = self.get_latest_epoch()
        if start_epoch > 0:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.result_dir, "optimizer_deepsdf_embedding.pth"), weights_only=True))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.result_dir, "scheduler_deepsdf_embedding.pth"), weights_only=True))
        
    def save_optimizer(self):
        torch.save(self.optimizer.state_dict(), os.path.join(self.result_dir, "optimizer_deepsdf_embedding.pth"))
        torch.save(self.scheduler.state_dict(), os.path.join(self.result_dir, "scheduler_deepsdf_embedding.pth"))
        
    def epoch_train(self, epoch):
        self.deepsdf_model.train()
        epoch_loss = 0
        for i, (points, sdfs, indices) in enumerate(self.deepsdf_train_dataloader):
            self.optimizer.zero_grad()
            points, sdfs = points.to(self.device), sdfs.to(self.device)
            sdfs = sdfs.view(-1, 1)
            
            latents = self.embeddings(indices.to(self.device))
            latents = latents.view(-1, self.latent_dim)
            
            # Forward pass
            sdf_preds = self.deepsdf_model(latents, points)
            
            # Compute loss
            loss = F.l1_loss(sdf_preds.view(-1), sdfs.view(-1), reduction="sum")
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        self.scheduler.step()
        return epoch_loss

if __name__ == "__main__":
    from model.DeepSDF import DeepSDF
    from data.dataset import ShapeNetSDF
    
    # Example usage
    shapenetsdf_path = ""
    deepsdf_model = DeepSDF(latent_dim=512, hidden_dim=512, n_hidden_layers=8)
    deepsdf_dataset = ShapeNetSDF(shapenetsdf_path)
    
    trainer = DeepSDFTrainer(
        deepsdf_model,
        deepsdf_dataset, 
        epochs=2000, batch_size=24, 
        results_dir="results_deepsdf_latent512_hidden512"
    )
