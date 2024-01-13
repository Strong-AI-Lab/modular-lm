from torch import nn
from transformers import Trainer


class RoutingTrainer(Trainer):
    def __init__(self, *args, routing_weight : float = 1.0, mutual_information_weight : float = 1.0, invariant_prediction_weight : float = 1.0, domain_prediction_weight : float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_weight = routing_weight
        self.mutual_information_weight = mutual_information_weight
        self.invariant_prediction_weight = invariant_prediction_weight
        self.domain_prediction_weight = domain_prediction_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute outputs and self-supervised loss
        s_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if isinstance(outputs, dict):
            # compute routing loss
            if outputs.get("routing_loss") is not None:
                r_loss = outputs.get("routing_loss")
            else:
                r_loss = 0.0
            # compute mutual information loss
            if outputs.get("mi_loss") is not None:
                mi_loss = outputs.get("mi_loss")
            else:
                mi_loss = 0.0
            # compute invariant loss
            if outputs.get("invariant_loss") is not None:
                i_loss = outputs.get("invariant_loss")
            else:
                i_loss = 0.0
            # compute domain loss
            if outputs.get("domain_loss") is not None:
                d_loss = outputs.get("domain_loss")
            else:
                d_loss = 0.0
        else:
            # compute routing loss
            if len(outputs) >= 4 and outputs[3] is not None:
                r_loss = outputs[3]
            else:
                r_loss = 0.0
            # compute mutual information loss
            if len(outputs) >= 5 and outputs[4] is not None:
                mi_loss = outputs[4]
            else:
                mi_loss = 0.0
            # compute invariant loss
            if len(outputs) >= 6 and outputs[5] is not None:
                i_loss = outputs[5]
            else:
                i_loss = 0.0
            # compute domain loss
            if len(outputs) >= 7 and outputs[6] is not None:
                d_loss = outputs[6]
            else:
                d_loss = 0.0

        loss = s_loss + self.routing_weight * r_loss + self.mutual_information_weight * mi_loss + self.invariant_prediction_weight * i_loss + self.domain_prediction_weight * d_loss

        return (loss, outputs) if return_outputs else loss