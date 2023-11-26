
from typing import Optional, Callable

from ..router.routing_strategy import RoutingStrategy
from ..loss.mi import batch_mutual_information_loss

import torch
from transformers import PreTrainedModel


class ModularModel(PreTrainedModel):

    def __init__(self, base_model_func : Callable, base_model_config, routing_strategy : RoutingStrategy, nb_modules : int = 4, invariant_weight : float = 1.0, hidden_dropout_prob : float = 0.1, **kwargs):
        super().__init__(base_model_config, **kwargs)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.invariant_weight = invariant_weight
        self.nb_modules = nb_modules

        self.domain_models = torch.nn.ModuleList()
        self.routing_strategy = routing_strategy

        self.router = base_model_func()
        self.invariant_model = base_model_func()
        for _ in range(nb_modules):
            self.domain_models.append(base_model_func())

    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # Compute router outputs
        router_outputs = self.router(
            input_ids.to(self.router.device),
            attention_mask=attention_mask.to(self.router.device),
            output_hidden_states=True,
        )
        domain_weights, routing_loss = self.compute_weights(router_outputs.hidden_states[-1])
        
        # Compute invariant outputs
        invariant_logits, invariant_probas = self.invariant_model(
            input_ids,
            attention_mask=attention_mask,
        )
        invariant_logits = self.dropout(invariant_logits)

        # Compute domain outputs
        domain_logits = []
        mi_loss = 0.0
        for domain_model_i in self.domain_models:
            domain_logits_i, domain_probas_i = domain_model_i(
                input_ids,
                attention_mask=attention_mask,
            )
            domain_logits_i = self.dropout(domain_logits_i)
            mi_loss += batch_mutual_information_loss(invariant_logits, domain_logits_i)
            domain_logits.append(domain_logits_i)

        aggregated_domain_logits = torch.sum(domain_weights * torch.stack(domain_logits, dim=0), dim=0)

        logits = self.invariant_weight * invariant_logits + aggregated_domain_logits
        probas = torch.sigmoid(logits)
        # probas = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probas": probas,
            "routing_loss": routing_loss,
            "mi_loss": mi_loss,
            "invariant_logits": invariant_logits,
            "domain_logits": aggregated_domain_logits,
        }
    

    def compute_weights(self, router_logits):
        """
        Differentiable or non-differentiable method providing the weight of each domain model. Based on quantization or clustering.
        """
        domain_weights, routing_loss = self.routing_strategy(router_logits)
        return domain_weights, routing_loss


