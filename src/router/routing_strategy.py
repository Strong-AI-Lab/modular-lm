
import torch


class RoutingStrategy(torch.nn.Module):
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.compute_routing(latents)
    
    def compute_routing(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract method")
    

class InputLevelRouting(RoutingStrategy):
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        weights, loss = self.compute_routing(latents)
        assert len(weights.shape) == 2 # [B x K]

        return weights, loss


class TokenLevelRouting(RoutingStrategy):
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        weights, loss = self.compute_routing(latents)
        assert len(weights.shape) == 3 # [B x L x K]
        
        return weights, loss