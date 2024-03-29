
import argparse
import numpy as np
import yaml
import datetime
import tqdm
from typing import Optional

from src.modular_lm.model.modular import ModularModel, ModularConfig
from src.modular_lm.trainer.routing_trainer import RoutingTrainer
from src.modular_lm.data.dataset import ProxyDataset

import torch
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset, Dataset
import evaluate




def main():
    parser = argparse.ArgumentParser(
        description="Run finetuning of given model for given dataset."
    )
    parser.add_argument("model_config")
    parser.add_argument("router_config")
    parser.add_argument("dataset_config")
    parser.add_argument("trainer_config")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)
    
    # Load router config file
    with open(args.router_config, "r") as router_config_file:
        router_config = yaml.safe_load(router_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)

    # Load trainer config file
    with open(args.trainer_config, "r") as trainer_config_file:
        trainer_config = yaml.safe_load(trainer_config_file)


    # Load metric
    metric = evaluate.load(trainer_config["metric"])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = predictions.reshape(-1).astype(dtype=np.int32)
        labels = labels.reshape(-1).astype(dtype=np.int32)
        return metric.compute(predictions=predictions, references=labels)


    # Load model
    if args.local_rank != -1:
        model_config["model_config"]["device_map"] = args.local_rank
    config = ModularConfig.from_pretrained(
                 "config/model/modular_pretrained/config.json",
                 base_model_path = model_config["model_path"], 
                 base_model_config = model_config["model_config"], 
                 routing_strategy_name = router_config["router_path"], 
                 routing_strategy_config = router_config["routing_strategy"],
                 routing_strategy_save = None if "save_path" not in router_config else router_config["save_path"],
                 is_peft = "peft" in trainer_config, 
                 peft_config = trainer_config["peft"],
                 **model_config["kwargs"])
    model = ModularModel(config=config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"], **model_config["tokenizer_config"])
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples, text_id : str = "text", label_id : str = "labels", dataset_id : str = "dataset", dataset_list : Optional[list] = None): # Non-batched tokenizer
        if label_id in examples:
            prompt = examples[text_id] + str(examples[label_id])
        else:
            prompt = examples[text_id]

        # Print warning if input is too long
        l = len(tokenizer.encode(prompt))
        if l > model_config["max_length"]:
            print(f"Input is too long: {l}. Truncating to {model_config['max_length']} tokens.")
        
        tokenized = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=model_config["max_length"])
        tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)

        if dataset_id in examples:
            if dataset_list is None:
                tokenized["dataset"] = torch.tensor(int(examples[dataset_id]), dtype=torch.int64)
            else:
                tokenized["dataset"] = torch.tensor(dataset_list.index(examples[dataset_id]), dtype=torch.int64)

        tokenized["labels"] = tokenized["input_ids"].clone()
        if label_id in examples:
            label_len = len(tokenizer.encode(str(examples[label_id]))) -1 # -1 because of the start token
            tokenized["labels"][:-label_len] = -100
        
        return tokenized

    # Load training dataset
    if "huggingface" in data_config and data_config["huggingface"]:
        dataset = load_dataset(data_config["dataset_path"], **data_config["dataset_config"])
        if "column_mappings" in data_config:
            ground_tokenize_function = lambda x : tokenize_function(x, **data_config["column_mappings"])
        else:
            ground_tokenize_function = tokenize_function
    elif "evals" in data_config and data_config["evals"]:
        dataset = Dataset.from_generator(ProxyDataset(data_config["dataset_path"], **data_config["dataset_config"]).generator)
        ground_tokenize_function = tokenize_function
        
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(ground_tokenize_function, batched=False)
    eval_dataset = dataset["test"]
    eval_dataset = eval_dataset.map(ground_tokenize_function, batched=False)


    # If using clustering router, learn clusters first
    if router_config["router_path"].endswith("LevelCluster") and "pretrain_cluster" in router_config and router_config["pretrain_cluster"]:
        embeddings = []
        batch = []
        cluster_batch_size = trainer_config["training_arguments"]["per_device_train_batch_size"] if "pretrain_batch_size" not in router_config else router_config["pretrain_batch_size"]
        for i in tqdm.trange(len(dataset["train"])):
            text = " ".join(dataset["train"][i]['text'])
            batch.append(text)

            if len(batch) == cluster_batch_size or i == len(dataset) - 1:
                input_ids = tokenizer(batch, return_tensors='pt', padding="max_length", truncation=True, max_length=model_config["max_length"])
                batch = []

                outputs = model.router(**input_ids, output_hidden_states=True)
                outputs = outputs.hidden_states[-1].detach()
                outputs = outputs.sum(dim=1).numpy()

                if np.isnan(outputs).any():
                    print("NaN encountered in the embeddings. Skipping this batch. This can happen when the batch size is too large.")
                else:
                    embeddings.extend(outputs)
        
        embeddings = torch.tensor(np.array(embeddings))
        model.routing_strategy.fit_cluster(embeddings)


    # Train model
    training_args = TrainingArguments(local_rank=args.local_rank, deepspeed=args.deepspeed, **trainer_config["training_arguments"])
    trainer = RoutingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        routing_weight=trainer_config["routing_weight"],
        mutual_information_weight=trainer_config["mutual_information_weight"],
    )
    trainer.train()


    # Save results
    save_path = f"fine-tuning-saves/fine-tuned-{model_config['model_name']}-{router_config['router_name']}-{data_config['dataset_name']}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    trainer.save_model(save_path)
    model.save_pretrained(save_path)
    model.config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    main()