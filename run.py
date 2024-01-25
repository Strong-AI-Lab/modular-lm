
import os
import argparse
import yaml
import tqdm
import re
import pandas as pd
import datetime
import json

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
            metrics = {"label" : label[i], "prediction" : prediction[i]}
            for metric_name, metric_function in METRICS.items():
                metrics[metric_name] = metric_function(prediction[i], label[i])
            results.append(metrics)
        return results


    # Evaluate model
    nb_lines = len(loader) if args.limit is None else min(len(loader), int(args.limit))
    results = []
    for i, line in tqdm.tqdm(enumerate(loader), total=nb_lines):
        input, label = line["text"], line["labels"]
        input = tokenizer(input, return_tensors="pt", padding=True)["input_ids"]

        tokenized_response = model.generate(input, max_new_tokens=model_config["max_length"]-input.shape[1])
        tokenized_response = [tokenized_response[i][len(input[i]):] for i in range(len(tokenized_response))]
        response = tokenizer.batch_decode(tokenized_response, skip_special_tokens=True)
        
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


if __name__ == "__main__":
    main()