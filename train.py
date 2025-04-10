import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from trainer.DeepSDFTrainer import DeepSDFTrainer
from data.dataset import ShapeNetSDF
from model.DeepSDF import DeepSDF

from trainer.VCCNFTrainer import VCCNFTrainer
from data.dataset import ShapeNetVoxel32
from model.VCCNF import VCCNF

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
            "init_lr" : 1e-3
        }
    }

    deepsdf_trainer = DeepSDFTrainer(
        deepsdf_model_infos,
        deepsdf_dataset,
        epochs=2000, batch_size=24, 
        results_dir="results/results_deepsdf_latent512_hidden512_dropout02_v2",
        model_save_frequency=100
    )

    for e, losses in tqdm(deepsdf_trainer, total=deepsdf_trainer.epochs - deepsdf_trainer.get_latest_epoch()):
        writer.add_scalar("Loss/DeepSDF_train", losses[-1], e)
        writer.flush()
        
    return deepsdf_trainer.embeddings
        
def train_vccnf(writer : SummaryWriter, embedding):
    vccnf_model = VCCNF(latent_dim=512, hidden_dim=512, n_hidden_layers=8)
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
        embedding=embedding
    )

    for e, losses in tqdm(enumerate(vccnf_trainer), total=vccnf_trainer.epochs):
        writer.add_scalar("Loss/VCCNF_train", losses[-1], e)
        writer.flush()
            
if __name__ == "__main__":
    is_train_deepsdf = True
    is_train_vccnf = False
    
    writer = SummaryWriter()
    
    embeddings = train_deepsdf(writer)

    if is_train_vccnf:
        train_vccnf(writer, embeddings)
    
    writer.close()
