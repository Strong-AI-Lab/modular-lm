from torch import nn
from transformers import Trainer


class RoutingTrainer(Trainer):
    def __init__(self, *args, routing_weight : float = 1.0, mutual_information_weight : float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_weight = routing_weight
        self.mutual_information_weight = mutual_information_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute outputs and self-supervised loss
        s_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # compute routing loss
        if outputs.get("routing_loss") is None:
            r_loss = 0.0
        else:
            r_loss = outputs.get("routing_loss")

        # compute mutual information loss
        if outputs.get("mi_loss") is None:
            mi_loss = 0.0
        else:
             mi_loss = outputs.get("mi_loss")

        loss = s_loss + self.routing_weight * r_loss + self.mutual_information_weight * mi_loss

        return (loss, outputs) if return_outputs else loss