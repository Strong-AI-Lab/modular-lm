
import torch


def _mutual_random_reduce_func(logits_p : torch.Tensor, logits_q: torch.Tensor, remaining_dim : int): # convert tensors of size [B x C] to [B x D] with D << C, select random subset of the dimensions
    dim_idx = torch.randperm(logits_p.shape[-1])[:remaining_dim]
    reduced_logits_p = logits_p[...,dim_idx]
    reduced_logits_q = logits_q[...,dim_idx]
    return reduced_logits_p, reduced_logits_q


def _mutual_max_reduce_func(logits_p : torch.Tensor, logits_q: torch.Tensor, remaining_dim : int): # convert tensors of size [B x C] to [B x D] with D << C, select dimensions with highest probability
    nb_dim_p = remaining_dim // 2
    nb_dim_q = remaining_dim - nb_dim_p

    dim_idx_p = torch.topk(logits_p, nb_dim_p).indices
    dim_idx_q = torch.topk(logits_q, nb_dim_q).indices
    dim_idx = torch.cat([dim_idx_p, dim_idx_q], dim=-1)
    dim_id0 = torch.arange(logits_p.shape[0]).unsqueeze(-1).repeat(1, remaining_dim)

    reduced_logits_p = logits_p[dim_id0, dim_idx]
    reduced_logits_q = logits_q[dim_id0, dim_idx]
    return reduced_logits_p, reduced_logits_q


def _mutual_absmax_reduce_func(logits_p : torch.Tensor, logits_q: torch.Tensor, remaining_dim : int): # convert tensors of size [B x C] to [B x D] with D << C, select dimensions with highest probability
    nb_dim_p = remaining_dim // 2
    nb_dim_q = remaining_dim - nb_dim_p

    dim_idx_p = torch.topk(logits_p.abs(), nb_dim_p).indices
    dim_idx_q = torch.topk(logits_q.abs(), nb_dim_q).indices
    dim_idx = torch.cat([dim_idx_p, dim_idx_q], dim=-1)
    dim_id0 = torch.arange(logits_p.shape[0]).unsqueeze(-1).repeat(1, remaining_dim)

    reduced_logits_p = logits_p[dim_id0, dim_idx]
    reduced_logits_q = logits_q[dim_id0, dim_idx]
    return reduced_logits_p, reduced_logits_q


class MutualReduce(torch.nn.Module):
    def __init__(self, reduce_func, remaining_dim : int):
        super().__init__()
        self.reduce_func = reduce_func
        self.remaining_dim = remaining_dim

    def forward(self, logits_p : torch.Tensor, logits_q: torch.Tensor, *args, **kwargs):
        reduced_logits_p, reduced_logits_q = self.reduce_func(logits_p, logits_q, self.remaining_dim) # [B x C] -> [B x D]
        return reduced_logits_p, reduced_logits_q

def mutual_random_reduce(remaining_dim : int):
    return MutualReduce(_mutual_random_reduce_func, remaining_dim)

def mutual_max_reduce(remaining_dim : int):
    return MutualReduce(_mutual_max_reduce_func, remaining_dim)

def mutual_absmax_reduce(remaining_dim : int):
    return MutualReduce(_mutual_absmax_reduce_func, remaining_dim)


class MutualBatchNormReduce(MutualReduce):
    def __init__(self, reduce_func, remaining_dim : int):
        super().__init__(reduce_func, remaining_dim)
        self.bn = torch.nn.BatchNorm1d(remaining_dim)

    def forward(self, logits_p : torch.Tensor, logits_q: torch.Tensor, *args, **kwargs):
        reduced_logits_p, reduced_logits_q = super().forward(logits_p, logits_q, *args, **kwargs) # [B x C] -> [B x D]
        logits_pq = torch.cat([reduced_logits_p, reduced_logits_q], dim=0) # [B x D], [B x D] -> [2B x D]
        logits_pq = self.bn(logits_pq)
        reduced_logits_p, reduced_logits_q = torch.split(logits_pq, logits_p.shape[0], dim=0) # [2B x D] -> [B x D], [B x D]

        return reduced_logits_p, reduced_logits_q

def mutual_batchnorm_random_reduce(remaining_dim : int):
    return MutualBatchNormReduce(_mutual_random_reduce_func, remaining_dim)

def mutual_batchnorm_max_reduce(remaining_dim : int):
    return MutualBatchNormReduce(_mutual_max_reduce_func, remaining_dim)

def mutual_batchnorm_absmax_reduce(remaining_dim : int):
    return MutualBatchNormReduce(_mutual_absmax_reduce_func, remaining_dim)


REDUCTION_FUNCTIONS = {
    "random" : mutual_random_reduce,
    "max" : mutual_max_reduce,
    "absmax" : mutual_absmax_reduce,
    "batchnorm_random" : mutual_batchnorm_random_reduce,
    "batchnorm_max" : mutual_batchnorm_max_reduce,
    "batchnorm_absmax" : mutual_batchnorm_absmax_reduce,
}



def batch_mutual_information_loss(logits_p: torch.Tensor, logits_q: torch.Tensor):
    p_x = logits_p.softmax(dim=-1).mean(dim=0) # [B x C] -> [C], compute P(X)
    p_y = logits_q.softmax(dim=-1).mean(dim=0) # [B x C] -> [C], compute P(Y)
    p_x_p_y = torch.einsum("i,j->ij", p_x, p_y) # [C], [C] -> [C x C], compute outer product P(X) ⊗ P(Y)
    log_p_x_p_y = p_x_p_y.clamp(min=1e-6, max=1-1e-6).log()

    p_xy = torch.einsum("ij,ik->ijk", logits_p.softmax(dim=-1), logits_q.softmax(dim=-1)) # [B x C], [B x C] -> [B x C x C], compute ∑_S P(X|S) ⊗ P(Y|S) P(S) = ∑_S P(X,Y|S) P(S) = P(X,Y) P(S) for all samples S in batch B as X and Y are conditionally independent given S
    log_xy =  p_xy.mean(dim=0).clamp(min=1e-6, max=1-1e-6).log() # [B x C x C] -> [C x C], perform summation of above formula

    mi_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)(
        log_xy.view((-1)), # [C x C] -> [CC]
        log_p_x_p_y.view((-1)), # [C x C] -> [CC]
    ).sum()
    return mi_loss # loss can sometimes be > 1 due to clamp


    