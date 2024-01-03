
import os
from typing import Union, Optional
import numpy as np

from .routing_strategy import TokenLevelRouting, InputLevelRouting

import torch
import torch.nn.functional as F

from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.cluster._kmeans import _BaseKMeans
from sklearn.manifold import MDS
import joblib


CLUSTERING_ALGORITHMS = {
    "Bisecting K-Means": BisectingKMeans,
    "K-Means": KMeans,
}


class Cluster(torch.nn.Module):

    @staticmethod
    def load_cluster(save_path : str, num_embeddings: int):
        clustering_algorithm = joblib.load(save_path)
        return Cluster(clustering_algorithm, num_embeddings)

    def __init__(self, clustering_algorithm : Union[str,_BaseKMeans], num_embeddings: int, training_latents: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_embeddings = num_embeddings

        if isinstance(clustering_algorithm, str):
            self.clustering_algorithm = CLUSTERING_ALGORITHMS[clustering_algorithm](n_clusters=num_embeddings)
            if training_latents is not None:
                self.fit_cluster(training_latents)
        else:
            self.clustering_algorithm = clustering_algorithm

    def save_cluster(self, save_path : str):
        joblib.dump(self.clustering_algorithm, save_path)

    def fit_cluster(self, latents: torch.Tensor):
        data = latents.detach().cpu().numpy()
        self.clustering_algorithm.fit(data)




class TokenLevelCluster(Cluster, TokenLevelRouting):

    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latent_shape = latents.shape
        latents = latents.view(-1, latents.shape[-1]) # [B x L x D] -> [BL x D]
        data = latents.detach().cpu().numpy() # non differentiable operation
        labels = self.clustering_algorithm.predict(data)
        labels = F.one_hot(torch.tensor(labels, device=latents.device, dtype=torch.int64), num_classes=self.num_embeddings).float() # non differentiable operation
        labels = labels.view(latent_shape[0], latent_shape[1], self.num_embeddings) # [BL x K] -> [B x L x K]
        labels = labels.permute(0, 2, 1) # [B x L x K] -> [B x K x L]

        return labels , None

    def save_strategy(self, path: str):
        self.save_cluster(os.path.join(path, "clusters.sav"))

    def load_strategy(self, path: str):
        self.clustering_algorithm = Cluster.load_cluster(os.path.join(path, "clusters.sav"), self.num_embeddings).clustering_algorithm
    

class DiffTokenLevelCluster(TokenLevelCluster):

    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.view(-1, latents.shape[-1]) # [B x L x D] -> [BL x D]
        centers = self.clustering_algorithm.cluster_centers_ # [K x D]
        centers = torch.tensor(centers, device=latents.device)

        distances = torch.cdist(latents, centers)
        distances = distances.view(latents.shape[0], latents.shape[1], self.num_embeddings)

        return distances, None
    



class InputLevelCluster(Cluster, InputLevelRouting):
        
    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.sum(dim=1) # [B x L x D] -> [B x D]
        data = latents.detach().cpu().numpy() # non differentiable operation
        labels = self.clustering_algorithm.predict(data)
        labels = F.one_hot(torch.tensor(labels, device=latents.device, dtype=torch.int64), num_classes=self.num_embeddings).float() # non differentiable operation

        return labels , None

    def save_strategy(self, path: str):
        self.save_cluster(os.path.join(path, "clusters.sav"))

    def load_strategy(self, path: str):
        self.clustering_algorithm = Cluster.load_cluster(os.path.join(path, "clusters.sav"), self.num_embeddings).clustering_algorithm
    

class DiffInputLevelCluster(InputLevelCluster):

    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.sum(dim=1) # [B x L x D] -> [B x D]
        centers = self.clustering_algorithm.cluster_centers_ # [K x D]
        centers = torch.tensor(centers, device=latents.device)

        distances = torch.cdist(latents, centers)

        return distances, None
    

class MDSInputLevelCluster(Cluster, InputLevelRouting):

    def __init__(self, clustering_algorithm : Union[str,_BaseKMeans], num_embeddings: int, training_latents: Optional[torch.Tensor] = None, reduction_components: Optional[int] = None, reduction_training_memory_size: Optional[int] = None):
        if reduction_components is None:
            reduction_components = num_embeddings
        self.reduction_components = reduction_components

        self.reduction_training_memory_size = reduction_training_memory_size
        self.reduction_training_memory = None

        self.dim_reduction = MDS(n_components=self.reduction_components)

        super().__init__(clustering_algorithm, num_embeddings, training_latents)
        
    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.sum(dim=1) # [B x L x D] -> [B x D]
        data = latents.detach().cpu().numpy() # non differentiable operation

        data = np.concatenate((self.reduction_training_memory, data), axis=0)
        data = self.dim_reduction.fit_transform(data.astype(np.float64))[self.reduction_training_memory_size:]

        labels = self.clustering_algorithm.predict(data)
        labels = F.one_hot(torch.tensor(labels, device=latents.device, dtype=torch.int64), num_classes=self.num_embeddings).float() # non differentiable operation

        return labels , None

    def save_strategy(self, path: str):
        self.save_cluster(os.path.join(path, "clusters.sav"))
        np.save(os.path.join(path, "reduction_training_memory.npy"), self.reduction_training_memory)

    def load_strategy(self, path: str):
        self.clustering_algorithm = Cluster.load_cluster(os.path.join(path, "clusters.sav"), self.num_embeddings).clustering_algorithm
        self.reduction_training_memory = np.load(os.path.join(path, "reduction_training_memory.npy"))

        if self.reduction_training_memory_size is None or self.reduction_training_memory_size > len(self.reduction_training_memory):
            self.reduction_training_memory_size = len(self.reduction_training_memory)
        else:
            self.reduction_training_memory = self.reduction_training_memory[:self.reduction_training_memory_size]
        
    def fit_cluster(self, latents: torch.Tensor):
        data = latents.detach().cpu().numpy()
        data = data[np.random.permutation(data.shape[0])] # shuffle data to avoid bias in the reduction

        if self.reduction_training_memory_size is None or self.reduction_training_memory_size > len(data):
            self.reduction_training_memory_size = len(data)
        self.reduction_training_memory = data[:self.reduction_training_memory_size]

        data = self.dim_reduction.fit_transform(data.astype(np.float64))
        self.clustering_algorithm.fit(data)