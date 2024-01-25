
import torch
from torch import nn
from transformers import Trainer, AutoTokenizer
import wandb


class RoutingTrainer(Trainer):
    def __init__(self, *args, routing_weight : float = 1.0, mutual_information_weight : float = 1.0, invariant_prediction_weight : float = 1.0, domain_prediction_weight : float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_weight = routing_weight
        self.mutual_information_weight = mutual_information_weight
        self.invariant_prediction_weight = invariant_prediction_weight
        self.domain_prediction_weight = domain_prediction_weight
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.base_model_path)

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
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "supervised_loss": s_loss.item() if isinstance(s_loss, torch.Tensor) else s_loss,
            "routing_loss": (self.routing_weight * r_loss).item() if isinstance(r_loss, torch.Tensor) else (self.routing_weight * r_loss),
            "mutual_information_loss": (self.mutual_information_weight * mi_loss).item() if isinstance(mi_loss, torch.Tensor) else (self.mutual_information_weight * mi_loss),
            "invariant_loss": (self.invariant_prediction_weight * i_loss).item() if isinstance(i_loss, torch.Tensor) else (self.invariant_prediction_weight * i_loss),
            "domain_loss": (self.domain_prediction_weight * d_loss).item() if isinstance(d_loss, torch.Tensor) else (self.domain_prediction_weight * d_loss),
        }
        if logits is not None:
            logs["logits_amplitude"] = logits.abs().mean().item()
        if invariant_logits is not None:
            logs["invariant_logits_amplitude"] = invariant_logits.abs().mean().item()
        if domain_logits is not None:
            logs["domain_logits_amplitude"] = domain_logits.abs().mean().item()

        self.log(logs)

        if "wandb" in self.args.report_to and "labels" in inputs:
            labels_index = (inputs["labels"] != -100).sum(dim=-1).tolist()
            labels = [label[-labels_index[i]:] for i, label in enumerate(inputs["labels"].tolist())]
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)

            if logits is not None:
                predictions = [logits[i,-1-labels_index[i]:-1].argmax(dim=-1) for i in range(logits.shape[0])] # shift logits to the left by one token to get the prediction
                predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
            else:
                predictions = [""] * len(labels)
            if invariant_logits is not None:
                invariant_predictions = [invariant_logits[i,-1-labels_index[i]:-1].argmax(dim=-1) for i in range(invariant_logits.shape[0])]
                invariant_predictions = self.tokenizer.batch_decode(invariant_predictions, skip_special_tokens=False)
            else:
                invariant_predictions = [""] * len(labels)
            if domain_logits is not None:
                domain_predictions = [domain_logits[i,-1-labels_index[i]:-1].argmax(dim=-1) for i in range(domain_logits.shape[0])]
                domain_predictions = self.tokenizer.batch_decode(domain_predictions, skip_special_tokens=False)
            else:
                domain_predictions = [""] * len(labels)

            table = wandb.Table(
                columns=["labels", "prediction", "invariant prediction", "domain prediction"], 
                data=[list(x) for x in zip(labels, predictions, invariant_predictions, domain_predictions)]
            )
            wandb.log({"predictions": table})

        if "wandb" in self.args.report_to and "dataset" in inputs:
            dataset = inputs["dataset"].tolist()

            if isinstance(outputs, dict):
                domain_weights = outputs.get("domain_weights")
            elif len(outputs) >= 10:
                domain_weights = outputs[9]
            else:
                domain_weights = None

            if domain_weights is not None:
                table = wandb.Table(
                    columns=["dataset"] + [f"domain weight_{j}" for j in range(domain_weights.shape[1])], 
                    data=[[str(dataset[i])] + [str(dw) for dw in domain_weights[i].tolist()] for i in range(domain_weights.shape[0])]
                )
                wandb.log({"domain weights": table})


        return (loss, outputs) if return_outputs else loss