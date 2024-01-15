
import os
from typing import Optional, Callable, Union, List, Tuple
from dataclasses import dataclass

from ..router.loader import load_router
from ..loss.mi import batch_mutual_information_loss, mutual_random_reduce, mutual_max_reduce

import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from transformers.utils import ModelOutput
from peft import get_peft_config, get_peft_model, prepare_model_for_kbit_training, PeftModel


class ModularConfig(PretrainedConfig):
    model_type = "modular"
    keys_to_ignore_at_inference = ["past_key_values"]

    def _log_warnings(
                self,
                base_model_path : Optional[str] = None, 
                base_model_config : Optional[dict] = None, 
                router_path : Optional[str] = None, 
                router_config : Optional[dict] = None, 
                invariant_model_path : Optional[str] = None, 
                invariant_model_config : Optional[dict] = None,  
                routing_strategy_name : Optional[str] = None,
                routing_strategy_config : Optional[dict] = None,
                is_peft : Union[bool,str,List[str]] = True, 
                peft_config : Optional[dict] = None):
        
        if base_model_path is not None and base_model_config is None:
            print("Warning! `base_model_config` must be provided if `base_model_path` is given.")
        if router_path is not None and router_config is None:
            print("Warning! `router_config` must be provided if `router_path` is given.")
        if invariant_model_path is not None and invariant_model_config is None:
            print("Warning! `invariant_model_config` must be provided if `invariant_model_path` is given.")
        if routing_strategy_name is None or routing_strategy_config is None:
            print("Warning! `routing_strategy_name` and `routing_strategy_config` must be provided.")
        if is_peft and peft_config is None:
            print("Warning! `peft_config` must be provided if `is_peft` is True.") 

    def _update_modules_from_base(
                 self,
                 router_path : Optional[str] = None, 
                 router_config : Optional[dict] = None, 
                 invariant_model_path : Optional[str] = None, 
                 invariant_model_config : Optional[dict] = None,  
                **kwargs):       

        if router_path is None: # if router path and config are not provided, use base model path and config
            self.router_path = self.base_model_path
            self.router_config = self.base_model_config
        else:
            self.router_path = router_path
            self.router_config = router_config

        if invariant_model_path is None:
            self.invariant_model_path = self.base_model_path
            self.invariant_model_config = self.base_model_config
        else:
            self.invariant_model_path = invariant_model_path
            self.invariant_model_config = invariant_model_config

        self.domain_model_paths = []
        self.domain_model_configs = []
        for i in range(self.nb_modules):
            if f"domain_model_{i}_path" in kwargs:
                if f"domain_model_{i}_config" not in kwargs:
                    raise ValueError(f"`domain_model_{i}_config` must be provided if `domain_model_{i}_path` is given.")

                self.domain_model_paths.append(kwargs[f"domain_model_{i}_path"])
                self.domain_model_configs.append(kwargs[f"domain_model_{i}_config"])
            else:
                self.domain_model_paths.append(self.base_model_path)
                self.domain_model_configs.append(self.base_model_config)
        

    def __init__(
                 self,
                 base_model_path : Optional[str] = None, 
                 base_model_config : Optional[dict] = None, 
                 router_path : Optional[str] = None, 
                 router_config : Optional[dict] = None, 
                 invariant_model_path : Optional[str] = None, 
                 invariant_model_config : Optional[dict] = None,  
                 routing_strategy_name : Optional[str] = None,
                 routing_strategy_config : Optional[dict] = None,
                 routing_strategy_save : Optional[str] = None,
                 nb_modules : int = 4, 
                 invariant_weight : float = 1.0, 
                 hidden_dropout_prob : float = 0.1, 
                 mi_dim_reduction : int = 1024,
                 mi_dim_reduce_method : str = "max",
                 is_peft : Union[bool,str,List[str]] = True, 
                 peft_config : Optional[dict] = None,
                **kwargs):
        super().__init__(**kwargs)

        self.nb_modules = nb_modules
        self.invariant_weight = invariant_weight
        self.hidden_dropout_prob = hidden_dropout_prob
        self.mi_dim_reduction = mi_dim_reduction
        self.mi_dim_reduce_method = mi_dim_reduce_method
        self.is_peft = is_peft
        self.peft_config = peft_config
        self.base_model_path = base_model_path
        self.base_model_config = base_model_config
        self.routing_strategy_name = routing_strategy_name
        self.routing_strategy_config = routing_strategy_config
        self.routing_strategy_save = routing_strategy_save

        self._log_warnings(base_model_path, base_model_config, router_path, router_config, invariant_model_path, invariant_model_config, routing_strategy_name, routing_strategy_config, is_peft, peft_config)
        self._update_modules_from_base(router_path, router_config, invariant_model_path, invariant_model_config, **kwargs)

    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs):
        config = super(ModularConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        config._update_modules_from_base(**kwargs)
        
        return config


@dataclass
class ModularOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    probas: Optional[torch.FloatTensor] = None
    routing_loss: Optional[torch.FloatTensor] = None
    mi_loss: Optional[torch.FloatTensor] = None
    invariant_loss: Optional[torch.FloatTensor] = None
    domain_loss: Optional[torch.FloatTensor] = None
    invariant_logits: Optional[torch.FloatTensor] = None
    domain_logits: Optional[torch.FloatTensor] = None
    domain_weights: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class ModularModel(PreTrainedModel):

    @classmethod
    def load_base_model(cls, model_path : str, model_config : dict, is_peft : bool = True, peft_config : Optional[dict] = None):
        config = AutoConfig.from_pretrained(model_path, **model_config)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **model_config)
        if is_peft:
            peft_config = get_peft_config(peft_config)
            model = prepare_model_for_kbit_training(model) # Add this for using int8
            model = get_peft_model(model, peft_config) # Add this for using PEFT
        return model

    def __init__(self, config : PretrainedConfig):                
        super().__init__(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.invariant_weight = config.invariant_weight
        self.nb_modules = config.nb_modules
        self.mi_dim_reduction = config.mi_dim_reduction
        self.is_peft = config.is_peft

        if config.mi_dim_reduce_method.lower() == "max":
            self.mi_dim_reduction_method = mutual_max_reduce
        elif config.mi_dim_reduce_method.lower() == "random":
            self.mi_dim_reduction_method = mutual_random_reduce
        else:
            raise ValueError(f"Invalid mutual information dimension reduction method {config.mi_dim_reduce_method}.")


        self.domain_models = torch.nn.ModuleList()
        self.routing_strategy = load_router(config.routing_strategy_name, config.routing_strategy_config, config.routing_strategy_save)

        peft_modules = [False] * (self.nb_modules + 2)
        if not self.is_peft or (isinstance(self.is_peft, str) and self.is_peft.lower() == "none"):
            pass
        elif isinstance(self.is_peft, bool) or (isinstance(self.is_peft, str) and self.is_peft.lower() == "all"):
            peft_modules = [True] * (self.nb_modules + 2)
        elif isinstance(self.is_peft, list) and len(self.is_peft) == self.nb_modules + 2 and all(isinstance(module, bool) for module in self.is_peft):
            peft_modules = self.is_peft
        elif isinstance(self.is_peft, list):
            peft_modules = [False] * (self.nb_modules + 2)
            for module_name in self.is_peft:
                if module_name.lower() == "router":
                    peft_modules[0] = True
                elif module_name.lower() == "invariant":
                    peft_modules[1] = True
                else:
                    try:
                        module_index = int(module_name)
                        peft_modules[module_index + 2] = True
                    except ValueError:
                        raise ValueError(f"Invalid module name {module_name}.")
        
        self.router = ModularModel.load_base_model(config.router_path, config.router_config, peft_modules[0], config.peft_config)
        self.invariant_model = ModularModel.load_base_model(config.invariant_model_path, config.invariant_model_config, peft_modules[1], config.peft_config)
        for i in range(config.nb_modules):
            self.domain_models.append(ModularModel.load_base_model(config.domain_model_paths[i], config.domain_model_configs[i], peft_modules[i+2], config.peft_config))

    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            full_input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            dataset: Optional[torch.LongTensor] = None
            ):
        # if past_key_values is not None, estimate the division per modules
        if past_key_values is not None:
            nb_keys_per_module = len(past_key_values) // (self.nb_modules + 2)

        # Compute router outputs
        router_outputs = self.router(
            input_ids if full_input_ids is None else full_input_ids, # past_key_values cannot be used with router, so we need to pass the full input ids during generation
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        domain_weights, routing_loss = self.compute_weights(router_outputs.hidden_states[-1])

        # Compute invariant outputs
        if self.invariant_weight >= 1e-6: # compute invariant logits only if invariant weight is non-zero
            invariant_outputs = self.invariant_model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                position_ids=position_ids,
                past_key_values=None if past_key_values is None else past_key_values[nb_keys_per_module:2*nb_keys_per_module],
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            invariant_logits = self.dropout(invariant_outputs.logits)
            domain_device = invariant_logits.device
        else:
            invariant_logits = None
            domain_device = router_outputs.hidden_states[-1].device

        # Compute domain outputs
        domain_outputs = []
        domain_logits = []
        mi_loss = 0.0
        vocab_size = self.config.vocab_size
        for i, domain_model_i in enumerate(self.domain_models):
            domain_outputs_i = domain_model_i(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                position_ids=position_ids,
                past_key_values=None if past_key_values is None else past_key_values[(i+2)*nb_keys_per_module:(i+3)*nb_keys_per_module],
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            domain_logits_i = self.dropout(domain_outputs_i.logits)
            domain_logits_i = domain_logits_i.to(domain_device)

            if invariant_logits is not None: # if invariant weight is non-zero, compute mutual information loss
                reduced_invariant_logits, reduced_domain_logits_i = self.mi_dim_reduction_method(invariant_logits.view(-1, vocab_size), domain_logits_i.view(-1, vocab_size), self.mi_dim_reduction) # reduce the dimensionality of the logits to avoid memory overflow
                mi_loss += batch_mutual_information_loss(reduced_invariant_logits, reduced_domain_logits_i) # compute mutual information loss
            
            domain_outputs.append(domain_outputs_i)
            domain_logits.append(domain_logits_i)

        domain_logits = torch.stack(domain_logits, dim=1)
        domain_weights = domain_weights.view((domain_weights.size(0), domain_weights.size(1), 1 if len(domain_weights.shape) < 3 else domain_weights.size(2), 1)).to(domain_device)
        domain_probas = torch.softmax(domain_logits, dim=-1)
        domain_costs = torch.sum(domain_logits.exp(), dim=-1).unsqueeze(-1) # Compute cost of each domain model
        domain_logits = torch.log(((1 - domain_weights) / 2 + domain_probas * domain_weights).clamp(min=1e-6,max=1-1e-6)) + torch.log(domain_costs) # Multiply probas by weights (centered around zero) and convert back to logits, clamp to avoid NaNs
        aggregated_domain_logits = torch.sum(domain_logits, dim=1) # Sum weighted logits from all modules

        if invariant_logits is not None: # if invariant weight is non-zero, sum invariant logits with domain logits, otherwise only use domain logits
            stacked_logits = torch.stack([invariant_logits, aggregated_domain_logits], dim=1)
            probas = torch.softmax(stacked_logits, dim=-1)
            costs = torch.sum(stacked_logits.exp(), dim=-1).unsqueeze(-1)
            weights = torch.tensor([self.invariant_weight, 1 - self.invariant_weight], device=domain_device).view((1, 2, 1, 1))
            logits = torch.log(((1 - weights) / 2 + probas * weights).clamp(min=1e-6,max=1-1e-6)) + torch.log(costs)
            logits = torch.sum(logits, dim=1)
        else:
            logits = aggregated_domain_logits

        probas = torch.softmax(logits, dim=-1)

        losses = [None] * 3
        if labels is not None: # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1205
            for i, module_logits in enumerate([logits] + ([] if invariant_logits is None else [invariant_logits, aggregated_domain_logits])):
                # Shift so that tokens < n predict n
                shift_logits = module_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                losses[i] = loss_fct(shift_logits, shift_labels)
        
        loss = losses[0]
        invariant_loss = losses[1]
        domain_loss = losses[2]
            
        # If use_cache is set to True, return past_key_values from all modules. Used in generate() to avoid recomputing the past_key_values at each generation step
        output_past_key_values = None if not use_cache else router_outputs.past_key_values + ((None,) * len(router_outputs.past_key_values) if invariant_logits is None else invariant_outputs.past_key_values) + tuple(values for domain_outputs_i in domain_outputs for values in domain_outputs_i.past_key_values)
        
        # If output_hidden_states is set to True, return hidden states from all modules
        output_hidden_states = None if not output_hidden_states else router_outputs.hidden_states + ((None,) * len(router_outputs.hidden_states) if invariant_logits is None else invariant_outputs.hidden_states) + tuple(values for domain_outputs_i in domain_outputs for values in domain_outputs_i.hidden_states)
        
        # If output_attentions is set to True, return attentions from all modules
        output_attentions = None if not output_attentions else router_outputs.attentions + ((None,) * len(router_outputs.attentions) if invariant_logits is None else invariant_outputs.attentions) + tuple(values for domain_outputs_i in domain_outputs for values in domain_outputs_i.attentions)

        if not return_dict:
            output = (logits, probas, routing_loss, mi_loss, invariant_loss, domain_loss, invariant_logits, aggregated_domain_logits, domain_weights, output_past_key_values, output_hidden_states, output_attentions)
            return (loss,) + output if loss is not None else output

        return ModularOutput(
            logits=logits,
            probas=probas,
            routing_loss=routing_loss,
            mi_loss=mi_loss,
            invariant_loss=invariant_loss,
            domain_loss=domain_loss,
            invariant_logits=invariant_logits,
            domain_logits=aggregated_domain_logits,
            domain_weights=domain_weights,
            loss=loss,
            past_key_values=output_past_key_values,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
            )
        

    def compute_weights(self, router_logits):
        """
        Differentiable or non-differentiable method providing the weight of each domain model. Based on quantization or clustering.
        """
        domain_weights, routing_loss = self.routing_strategy(router_logits)
        return domain_weights, routing_loss
    


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs):

        os.makedirs(save_directory, exist_ok=True)
        routing_strategy_directory = os.path.join(save_directory, "routing_strategy")
        os.makedirs(routing_strategy_directory, exist_ok=True)
        router_directory = os.path.join(save_directory, "router")
        os.makedirs(router_directory, exist_ok=True)
        invariant_directory = os.path.join(save_directory, "invariant")
        os.makedirs(invariant_directory, exist_ok=True)
        domain_directories = []
        for i in range(self.nb_modules):
            domain_directory = os.path.join(save_directory, f"domain_{i}")
            os.makedirs(domain_directory, exist_ok=True)
            domain_directories.append(domain_directory)

        self.routing_strategy.save_strategy(routing_strategy_directory)

        self.router.save_pretrained(router_directory, is_main_process=is_main_process, state_dict=state_dict, save_function=save_function, push_to_hub=push_to_hub, max_shard_size=max_shard_size, safe_serialization=safe_serialization, variant=variant, token=token, save_peft_format=save_peft_format, **kwargs)
        self.invariant_model.save_pretrained(invariant_directory, is_main_process=is_main_process, state_dict=state_dict, save_function=save_function, push_to_hub=push_to_hub, max_shard_size=max_shard_size, safe_serialization=safe_serialization, variant=variant, token=token, save_peft_format=save_peft_format, **kwargs)
        for domain_model, domain_directory in zip(self.domain_models, domain_directories):
            domain_model.save_pretrained(domain_directory, is_main_process=is_main_process, state_dict=state_dict, save_function=save_function, push_to_hub=push_to_hub, max_shard_size=max_shard_size, safe_serialization=safe_serialization, variant=variant, token=token, save_peft_format=save_peft_format, **kwargs)
        

    @classmethod
    def remove_from_peft(cls, is_peft : Union[bool, list], module_name : str, nb_modules : int):
        if (isinstance(is_peft, bool) and is_peft) or (isinstance(is_peft, str) and is_peft.lower() == "all"):
            is_peft = [True] * (nb_modules + 2)
        
        if isinstance(is_peft, list):
            if module_name.lower() == "router":
                is_peft[0] = False
            elif module_name.lower() == "invariant":
                is_peft[1] = False
            else:
                try:
                    module_index = int(module_name)
                    is_peft[module_index + 2] = False
                except ValueError:
                    raise ValueError(f"Invalid module name {module_name}.")
                
        return is_peft


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        **kwargs):         

        # Load or build config  
        if config is None:
            config = ModularConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif not isinstance(config, PretrainedConfig):
            config = ModularConfig(**{**config, **kwargs})
        
        # If path contains modules saves, update config accordingly
        routing_strategy_directory = os.path.join(pretrained_model_name_or_path, "routing_strategy")
        if os.path.isdir(routing_strategy_directory): # load modules from save directory instead of base config if a save is provided
            config.routing_strategy_save = routing_strategy_directory

        router_directory = os.path.join(pretrained_model_name_or_path, "router")
        if os.path.isdir(router_directory):
            if os.path.isfile(os.path.join(router_directory, "adapter_config.json")): # if peft modules, load base config first (without peft) and adapt later 
                config.is_peft = cls.remove_from_peft(config.is_peft, "router", config.nb_modules)
            else: # otherwise, load router directly from save directory
                config.router_path = router_directory

        invariant_directory = os.path.join(pretrained_model_name_or_path, "invariant")
        if os.path.isdir(invariant_directory):
            if os.path.isfile(os.path.join(invariant_directory, "adapter_config.json")): # if peft modules, load base config first (without peft) and adapt later
                config.is_peft = cls.remove_from_peft(config.is_peft, "invariant", config.nb_modules)
            else: # otherwise, load invariant directly from save directory
                config.invariant_model_path = invariant_directory
        
        domain_directories = []
        for i in range(config.nb_modules):
            domain_directory = os.path.join(pretrained_model_name_or_path, f"domain_{i}")
            domain_directories.append(domain_directory)
            if os.path.isdir(domain_directory):
                if os.path.isfile(os.path.join(domain_directory, "adapter_config.json")): # if peft modules, load base config first (without peft) and adapt later
                    config.is_peft = cls.remove_from_peft(config.is_peft, str(i), config.nb_modules)
                else: # otherwise, load domain directly from save directory
                    config.domain_model_paths[i] = domain_directory

        # Load model
        model = cls(config)

        # If path contains PEFT adapter saves, load them
        if os.path.isfile(os.path.join(router_directory, "adapter_config.json")):
            model.router = prepare_model_for_kbit_training(model.router)
            model.router = PeftModel.from_pretrained(model.router, router_directory)
            model.router = model.router.merge_and_unload()

        if os.path.isfile(os.path.join(invariant_directory, "adapter_config.json")):
            model.invariant_model = prepare_model_for_kbit_training(model.invariant_model)
            model.invariant_model = PeftModel.from_pretrained(model.invariant_model, invariant_directory)
            model.invariant_model = model.invariant_model.merge_and_unload()

        for i in range(config.nb_modules):
            if os.path.isfile(os.path.join(domain_directories[i], "adapter_config.json")):
                model.domain_models[i] = prepare_model_for_kbit_training(model.domain_models[i])
                model.domain_models[i] = PeftModel.from_pretrained(model.domain_models[i], domain_directories[i])
                model.domain_models[i] = model.domain_models[i].merge_and_unload()

        return model
    
    

    def prepare_inputs_for_generation( # Taken from <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1227>
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        full_input_ids = None
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            full_input_ids = input_ids.clone()
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "full_input_ids": full_input_ids
            }
        )
        return model_inputs