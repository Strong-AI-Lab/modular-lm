
import torch


class WeightedLogitAggregator(torch.nn.Module):

    def __init__(self, eps : int = 1.0, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits : torch.Tensor, # [batch_size, nb_modules, seq_len, vocab_size]
        weights : torch.Tensor # [batch_size, nb_modules, seq_len, vocab_size]
    ):
        return _weighted_logit_aggregator_func(logits * self.eps, weights) / self.eps 


class BatchNormWeightedLogitAggregator(torch.nn.Module):
    
    def __init__(self, num_embeddings : int, eps : int = 1.0, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.eps = eps
        self.bn = torch.nn.BatchNorm1d(num_embeddings)

    def forward(
        self,
        logits : torch.Tensor, # [batch_size, nb_modules, seq_len, vocab_size]
        weights : torch.Tensor # [batch_size, nb_modules, seq_len, vocab_size]
    ):
        logits_shape = logits.shape
        logits = logits.view((-1, logits_shape[2], logits_shape[3])) # [batch_size * nb_modules, seq_len, vocab_size]
        logits = logits.permute(0, 2, 1) # [batch_size * nb_modules, vocab_size, seq_len]
        logits = self.bn(logits)
        logits = logits.permute(0, 2, 1)
        logits = logits.view(logits_shape)

        return _weighted_logit_aggregator_func(logits * self.eps, weights) / self.eps 




def _weighted_proba_aggregator_func(logits : torch.Tensor, weights : torch.Tensor):
    probas = torch.softmax(logits, dim=-1)
    costs = torch.sum(logits.exp(), dim=-1).unsqueeze(-1) # Compute cost of each domain model

    weighted_logits = torch.log(((1 - weights) / 2 + weights * probas).clamp(min=1e-6,max=1-1e-6)) + torch.log(costs) # Multiply probas by weights (centered around zero) and convert back to logits, clamp to avoid NaNs
    weighted_logits = torch.sum(weighted_logits, dim=1) # Sum weighted logits from all modules

    return weighted_logits


def _weighted_logit_aggregator_func(logits : torch.Tensor, weights : torch.Tensor):
    weighted_logits = weights * logits
    weighted_logits = torch.sum(weighted_logits, dim=1) # Sum weighted logits from all modules

    return weighted_logits
