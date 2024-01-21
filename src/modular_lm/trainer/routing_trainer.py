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

            # compute logits amplitude
            logits = outputs.get("logits")
            # compute invariant logits amplitude
            invariant_logits = outputs.get("invariant_logits")
            # compute domain logits amplitude
            domain_logits = outputs.get("domain_logits")
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
            
            # compute logits amplitude
            if len(outputs) >= 2:
                logits = outputs[1]
            # compute invariant logits amplitude
            if len(outputs) >= 8:
                invariant_logits = outputs[7]
            # compute domain logits amplitude
            if len(outputs) >= 9:
                domain_logits = outputs[8]

        loss = s_loss + self.routing_weight * r_loss + self.mutual_information_weight * mi_loss + self.invariant_prediction_weight * i_loss + self.domain_prediction_weight * d_loss
        
        logs = {
            "loss": loss.item(),
            "supervised_loss": s_loss.item(),
            "routing_loss": (self.routing_weight * r_loss).item(),
            "mutual_information_loss": (self.mutual_information_weight * mi_loss).item(),
            "invariant_loss": (self.invariant_prediction_weight * i_loss).item(),
            "domain_loss": (self.domain_prediction_weight * d_loss).item(),
        }
        if logits is not None:
            logs["logits_amplitude"] = logits.abs().mean().item()
        if invariant_logits is not None:
            logs["invariant_logits_amplitude"] = invariant_logits.abs().mean().item()
        if domain_logits is not None:
            logs["domain_logits_amplitude"] = domain_logits.abs().mean().item()

        self.log(logs)

        return (loss, outputs) if return_outputs else loss