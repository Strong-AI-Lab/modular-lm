
from .routing_strategy import TokenLevelRouting, InputLevelRouting

import torch
import torch.nn.functional as F


class TokenQuantizer(TokenLevelRouting):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 quantize_fn: str = "hard",
                 beta: float = 0.25,
                 **kwargs):
        super(TokenQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = torch.nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        if quantize_fn == "hard":
            self.quantize_fn = self._hard_quantize
        elif quantize_fn == "soft":
            self.quantize_fn = self._soft_quantize
        elif quantize_fn == "gumbel":
            self.quantize_fn = self._gumbel_quantize
        elif self.quantize_fn == "vector":
            self.quantize_fn = self._vector_quantize
        else:
            raise ValueError(f"Unknown quantize function: {quantize_fn}. Valid options are: 'hard', 'soft', 'gumbel', 'vector'.")

    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [B x L x D] -> [BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]
        # dist = dist.view((latents_shape[0], latents_shape[1], self.K))  # [B x L x K]
        
        return self.quantize_fn(dist, latents, latents_shape)
    
    def _hard_quantize(self, distances: torch.Tensor, latents: torch.Tensor, latents_shape: torch.Size) -> torch.Tensor:
        encoding_one_hot = F.one_hot(distances.argmin(dim=-1), num_classes=self.K).float() # non-differentiable operation
        return encoding_one_hot.view((latents_shape[0], latents_shape[1], self.K)), None  # [B x L x K]
    
    def _soft_quantize(self, distances: torch.Tensor, latents: torch.Tensor, latents_shape: torch.Size) -> torch.Tensor:
        encoding = distances.softmin(dim=-1)
        return encoding.view((latents_shape[0], latents_shape[1], self.K)), None  # [B x L x K]
    
    def _gumbel_quantize(self, distances: torch.Tensor, latents: torch.Tensor, latents_shape: torch.Size) -> torch.Tensor:
        encoding_one_hot = F.gumbel_softmax(distances, tau=1.0, hard=True)
        return encoding_one_hot.view((latents_shape[0], latents_shape[1], self.K)), None  # [B x L x K]
    
    def _vector_quantize(self, distances: torch.Tensor, latents: torch.Tensor, latents_shape: torch.Size) -> torch.Tensor:
        encoding_one_hot = F.one_hot(distances.argmin(dim=-1), num_classes=self.K).float() # non-differentiable operation

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL x D]
        quantized_latents = quantized_latents.view(latents_shape) # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        return encoding_one_hot.view((latents_shape[0], latents_shape[1], self.K)), vq_loss
    
    def save_strategy(self, path: str):
        torch.save(self.embedding.weight.data, path + "/embeddings.pt")

    def load_strategy(self, path: str):
        self.embedding.weight.data = torch.load(path + "/embeddings.pt")




class InputQuantizer(InputLevelRouting):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 quantize_fn: str = "hard",
                 **kwargs):
        super(InputQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim

        self.embedding = torch.nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        if quantize_fn == "hard":
            self.quantize_fn = self._hard_quantize
        elif quantize_fn == "soft":
            self.quantize_fn = self._soft_quantize
        elif quantize_fn == "gumbel":
            self.quantize_fn = self._gumbel_quantize
        else:
            raise ValueError(f"Unknown quantize function: {quantize_fn}. Valid options are: 'hard', 'soft', 'gumbel'.")

    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        latents_shape = latents.shape
        latents = latents.to(self.embedding.weight.device)
        flat_latents = latents.view(-1, self.D)  # [B x L x D] -> [BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]
        
        dist = dist.view((latents_shape[0], latents_shape[1], self.K))  # [B x L x K]
        dist = dist.sum(dim=1)  # [B x K]
        
        return self.quantize_fn(dist)
    
    def _hard_quantize(self, distances: torch.Tensor) -> torch.Tensor:
        return F.one_hot(distances.argmin(dim=-1), num_classes=self.K).float(), None # non-differentiable operation
    
    def _soft_quantize(self, distances: torch.Tensor) -> torch.Tensor:
        return distances.softmin(dim=-1), None
    
    def _gumbel_quantize(self, distances: torch.Tensor) -> torch.Tensor:
        return F.gumbel_softmax(distances, tau=1.0, hard=True), None
    
    def save_strategy(self, path: str):
        torch.save(self.embedding.weight.data, path + "/embeddings.pt")

    def load_strategy(self, path: str):
        self.embedding.weight.data = torch.load(path + "/embeddings.pt")
    
