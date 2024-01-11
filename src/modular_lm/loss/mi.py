
import torch


def mutual_random_reduce(logits_p : torch.Tensor, logits_q: torch.Tensor, remaining_dim : int): # convert tensors of size [B x C] to [B x D] with D << C, select random subset of the dimensions
    dim_idx = torch.randperm(logits_p.shape[-1])[:remaining_dim]
    reduced_logits_p = logits_p[...,dim_idx]
    reduced_logits_q = logits_q[...,dim_idx]
    return reduced_logits_p, reduced_logits_q


def mutual_max_reduce(logits_p : torch.Tensor, logits_q: torch.Tensor, remaining_dim : int): # convert tensors of size [B x C] to [B x D] with D << C, select dimensions with highest probability
    nb_dim_p = remaining_dim // 2
    nb_dim_q = remaining_dim - nb_dim_p

    dim_idx_p = torch.topk(logits_p, nb_dim_p).indices
    dim_idx_q = torch.topk(logits_q, nb_dim_q).indices
    dim_idx = torch.cat([dim_idx_p, dim_idx_q], dim=-1)
    dim_id0 = torch.arange(logits_p.shape[0]).unsqueeze(-1).repeat(1, remaining_dim)

    reduced_logits_p = logits_p[dim_id0, dim_idx]
    reduced_logits_q = logits_q[dim_id0, dim_idx]
    return reduced_logits_p, reduced_logits_q



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
    return mi_loss


    