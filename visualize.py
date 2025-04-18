import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import trimesh
from skimage import measure
import os
from math import sqrt

from model.DeepSDF import DeepSDF
from model.VCCNF import VCCNF

class Visualizer:
    def __init__(self, cnf_encoder : VCCNF, sdf_decoder : DeepSDF, latent_vecs : nn.Embedding):
        self.cnf_encoder = cnf_encoder
        self.sdf_decoder = sdf_decoder
        self.latent_dim = self.sdf_decoder.latent_dim

        self.latent_vecs = latent_vecs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sdf_decoder.to(self.device)

    def get_sdf_for_marching_cube(self, latent_vec):
        # Initialize to mesh grid points
        x, y, z = np.mgrid[-1 : 1 : 0.02, -1 : 1 : 0.02, -1 : 1 : 0.02]
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = torch.tensor(z)

        # Change the mesh grid points from 100x100x100x3 to 1000000x3
        points = torch.stack((x, y, z), dim=3).view(-1, 3).float()
        points = points.to(self.device)

        # Split the points into 4 seperate splits to prevent running out of memory
        points_splits = points.split(250000)

        latent_code = latent_vec
        latent_code = latent_code.to(self.device)

        sdfs = torch.zeros(0, device="cpu")
        for points_split in points_splits:
            points_split = points_split.view(1, 250000, 3)
            sdfs = torch.cat((sdfs, torch.clamp(self.sdf_decoder(latent_code, points_split).cpu(), -0.1, 0.1)))

        sdfs = sdfs.view((100, 100, 100))
        sdfs = sdfs.numpy()
        return sdfs

    def generate_sdf_objs(self, path, index):
        self.sdf_decoder.eval()
        with torch.no_grad():
            # Get the sdf values from model
            sdfs = self.get_sdf_for_marching_cube(self.latent_vecs(torch.tensor([index]).cuda()))
            
            print(f"Volume range: {np.min(sdfs)} to {np.max(sdfs)}")

            # Marching Cube
            verts, faces, normals, _ = measure.marching_cubes(sdfs, 0)
            
            # Output the result into mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
            
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, f"mesh_reconstruct_model_{index}.obj"), "w") as f:
                mesh.export(f, "obj")
                
    def generate_sdf_objs_with_full_network(self, mesh_path, voxel_model):
        self.cnf_encoder.eval()
        self.sdf_decoder.eval()
        
        self.cnf_encoder = self.cnf_encoder.to(self.device)
        self.sdf_decoder = self.sdf_decoder.to(self.device)
        with torch.no_grad():
            gaussian_latent = (torch.randn((1, self.latent_dim)) / sqrt(self.latent_dim)).to(self.device)
            latent_code = self.cnf_encoder(voxel_model.view(1, 32, 32, 32).to(self.device), gaussian_latent)
            sdfs = self.get_sdf_for_marching_cube(latent_code)
        
            print(f"Volume range: {np.min(sdfs)} to {np.max(sdfs)}")

            # Marching Cube
            verts, faces, normals, _ = measure.marching_cubes(sdfs, 0)
            
            # Output the result into mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
            
            if not os.path.exists(os.path.split(mesh_path)[0]):
                os.makedirs(os.path.split(mesh_path)[0])
            with open(mesh_path, "w") as f:
                mesh.export(f, "obj")
            