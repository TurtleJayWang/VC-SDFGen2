import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from trainer.DeepSDFTrainer import DeepSDFTrainer
from data.dataset import ShapeNetSDF
from model.DeepSDF import DeepSDF

from trainer.VCCNFTrainer import VCCNFTrainer
from data.dataset import ShapeNetVoxel32
from model.VCCNF import VCCNF

from visualize import Visualizer
import random

import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import kagglehub
shapenetsdf_path = kagglehub.dataset_download("turtlejaywang/shapenetsdf")
shapenetvoxel32_path = kagglehub.dataset_download("turtlejaywang/shapenetvoxel64")

def train_deepsdf(writer : SummaryWriter):
    deepsdf_model = DeepSDF(latent_dim=512, hidden_dim=512, n_hidden_layers=8)
    deepsdf_dataset = ShapeNetSDF(shapenetsdf_path)

    deepsdf_model_infos = {
        "deepsdf" : {
            "model" : deepsdf_model,
            "init_lr" : 5e-4
        }
    }

    deepsdf_trainer = DeepSDFTrainer(
        deepsdf_model_infos,
        deepsdf_dataset,
        epochs=2000, batch_size=24,
        results_dir="results/results_deepsdf_latent512_hidden512_dropout02_v1",
        model_save_frequency=100
    )

    for e, losses in tqdm(deepsdf_trainer):
        writer.add_scalar("Loss/DeepSDF_train", losses[-1], e)
        writer.flush()
        
    return deepsdf_trainer
        
def train_vccnf(writer : SummaryWriter, embeddings):
    vccnf_model = VCCNF(latent_dim=512, hidden_dim=512, voxel_grid_size=32, voxel_latent_dim=512)
    shapenetvoxel32 = ShapeNetVoxel32(shapenetvoxel32_path)

    vccnf_model_info = {
        "vccnf" : {
            "model" : vccnf_model,
            "init_lr" : 1e-3
        }
    }

    vccnf_trainer = VCCNFTrainer(
        vccnf_model_info,
        shapenetvoxel32,
        epochs=2000, batch_size=24, 
        results_dir="results/results_vccnf",
        embeddings=embeddings
    )

    for e, losses in tqdm(vccnf_trainer):
        writer.add_scalar("Loss/VCCNF_train", losses[-1], e)
        writer.flush()
        
if __name__ == "__main__":
    is_train_deepsdf = True
    is_train_vccnf = True
    is_visualize = False
    
    writer = SummaryWriter()
    
    deepsdf_trainer = train_deepsdf(writer)
    embeddings = deepsdf_trainer.embeddings
    deepsdf_model = deepsdf_trainer.deepsdf_model
    
    if is_visualize:
        visualizer = Visualizer(deepsdf_model, embeddings)
        for i in range(0, 10):
            visualizer.generate_sdf_objs("results/model_recontruct/phase1", i)

    if is_train_vccnf:
        train_vccnf(writer, embeddings)
    
    writer.close()
