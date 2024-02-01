
import os
import argparse
import yaml
import tqdm
import re
import pandas as pd
import datetime
import json

import torch
from torch.utils.data import DataLoader

from src.modular_lm.model.modular import ModularModel, ModularConfig
from src.modular_lm.data.dataset import ProxyDataset

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset
from peft import PeftModel



METRICS = {
    "accuracy" : lambda x, y: float(x.lower() == y.lower()),
    "partial_accuracy" : lambda x, y: float(re.search(r'\b({})\b'.format(x.lower()), y.lower()) is not None),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run finetuning of given model for given dataset."
    )
    parser.add_argument("model_config")
    parser.add_argument("dataset_config")
    parser.add_argument("--module", type=str, default=None, help="If specified, evaluate a specific module within the Modular model. Options: 'router', 'invariant', 'domain_{i}' Provide the module number you want instead of the `i`.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()
    
    # Load model config file
    with open(args.model_config, "r") as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    # Load dataset config file
    with open(args.dataset_config, "r") as data_config_file:
        data_config = yaml.safe_load(data_config_file)


    # Load model
    if args.module:
        if not args.module in os.listdir(model_config["model_path"]):
            raise ValueError(f"Module {args.module} not found in model {model_config['model_path']}")
        if not args.module in ["router", "invariant"] and not args.module.startswith("domain_"):
            raise ValueError(f"Module {args.module} is not an allowed module. Options: 'router', 'invariant', 'domain_{i}' Provide the module number you want instead of the `i`.")

        module_path = os.path.join(model_config["model_path"], args.module)
        if "adapter_config.json" in os.listdir(module_path):
            with open(os.path.join(model_config["model_path"],"config.json"), "r") as base_config_file:
                base_weights = json.load(base_config_file)["base_model_path"]
            config = AutoConfig.from_pretrained(base_weights)
            model = AutoModelForCausalLM.from_pretrained(base_weights, config=config)

            model = PeftModel.from_pretrained(model, module_path)
            model = model.merge_and_unload()
        else:
            config = AutoConfig.from_pretrained(module_path)
            model = AutoModelForCausalLM.from_pretrained(module_path, config=config)
    else:
        config = ModularConfig.from_pretrained(model_config["model_path"], **model_config["model_config"])
        model = ModularModel.from_pretrained(model_config["model_path"], config=config, **model_config["model_config"])

    if args.gpu is not None:
        model = model.to(args.gpu)
    
    model.eval()
    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"], **model_config["tokenizer_config"])
    tokenizer.pad_token = tokenizer.eos_token
    

    # Load evaluation dataset
    if "huggingface" in data_config and data_config["huggingface"]:
        dataset = load_dataset(data_config["dataset_path"], **data_config["dataset_config"])
    elif "evals" in data_config and data_config["evals"]:
        dataset = Dataset.from_generator(ProxyDataset(data_config["dataset_path"], **data_config["dataset_config"]).generator)
        
    loader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size)


    # Evaluation metric
    def compute_metrics(prediction, label):
        if isinstance(label, str):
            label = [label]
            prediction = [prediction]

        results = []
        for i in range(len(label)):

            metrics = {"label" : str(label[i]), "prediction" : prediction[i]}
            for metric_name, metric_function in METRICS.items():
                metrics[metric_name] = metric_function(prediction[i], str(label[i]))
            results.append(metrics)
        return results


    # Evaluate model
    if "column_mappings" in data_config:
        if "text_id" in data_config["column_mappings"]:
            text_id = data_config["column_mappings"]["text_id"]
        if "label_id" in data_config["column_mappings"]:
            label_id = data_config["column_mappings"]["label_id"]
    else:
        text_id = "text"
        label_id = "labels"
    nb_lines = len(loader) if args.limit is None else min(len(loader), int(args.limit))
    results = []
    for i, line in tqdm.tqdm(enumerate(loader), total=nb_lines):
        input, label = line[text_id], line[label_id]
        input = tokenizer(input, return_tensors="pt", padding=True)["input_ids"]

        if args.gpu is not None:
            input = input.to(args.gpu)

        tokenized_response = model.generate(input, max_new_tokens=model_config["max_length"]-input.shape[1])
        tokenized_response = [tokenized_response[i][len(input[i]):] for i in range(len(tokenized_response))]
        response = tokenizer.batch_decode(tokenized_response, skip_special_tokens=True)
        
        if isinstance(label, torch.Tensor):
            label = label.tolist()

        results += compute_metrics(response, label)
        
        if args.limit is not None and i >= args.limit:
            break

    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs("inference-results", exist_ok=True)
    save_path = f"inference-results/results-{model_config['model_name']}{'' if args.module is None else f'-{args.module}'}-{data_config['dataset_name']}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    df.to_csv(save_path, index=False)

    # Print summary
    print(f"Results saved to {save_path}")
    print(*[f"{metric} : {df[metric].mean()}" for metric in METRICS.keys()], sep="\n")

    if df.label.unique().tolist() == [0, 1] or df.label.unique().tolist() == ['0', '1']: # compute binary metrics if binary output
        df.prediction[(df.prediction!=0) & (df.prediction!=1) & (df.prediction!='0') & (df.prediction!='1')] =  0
        df.label = df.label.map(int)
        df.prediction = df.prediction.map(int)

        true_positives = len(df[(df.label == 1) & (df.prediction == 1)])
        true_negatives = len(df[(df.label == 0) & (df.prediction == 0)])
        false_positives = len(df[(df.label == 0) & (df.prediction == 1)])
        false_negatives = len(df[(df.label == 1) & (df.prediction == 0)])
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        total = len(df)

        print(f"True positives : {true_positives}")
        print(f"True negatives : {true_negatives}")
        print(f"False positives : {false_positives}")
        print(f"False negatives : {false_negatives}")
        print(f"F1 : {f1}")


if __name__ == "__main__":
    main()