
import argparse
import numpy as np
import yaml
import datetime

from src.model.modular import ModularModel
from src.router.loader import load_router
from src.trainer.routing_trainer import RoutingTrainer

from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from datasets import load_dataset
import evaluate
from peft import get_peft_config, get_peft_model, prepare_model_for_int8_training




def main():
    parser = argparse.ArgumentParser(
        description="Run finetuning of given model for given dataset."
    )
    parser.add_argument("model_config")
    parser.add_argument("router_config")
    parser.add_argument("dataset_config")
    parser.add_argument("trainer_config")
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
    def load_base_model():
        base_model = AutoModelForCausalLM.from_pretrained(model_config["model_path"], **model_config["model_config"])
        peft_config = get_peft_config(trainer_config["peft"])
        base_model = prepare_model_for_int8_training(base_model) # Add this for using int8
        base_model = get_peft_model(base_model, peft_config) # Add this for using PEFT
        return base_model

    model = ModularModel(
        base_model_func=load_base_model,
        base_model_config=PretrainedConfig.from_pretrained(model_config["model_path"], model_config["model_config"]),
        routing_strategy=load_router(router_config["router_path"], router_config["routing_strategy"]),
        **model_config["kwargs"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"], **model_config["tokenizer_config"])
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(" ".join(examples['answers']['text']), padding="max_length", truncation=True, max_length=model_config["max_length"])


    # Load evaluation dataset
    dataset = load_dataset(data_config["dataset_path"], **data_config["dataset_config"])
    dataset = dataset.train_test_split(test_size=0.8, shuffle=True, seed=42)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(tokenize_function, batched=False)
    eval_dataset = dataset["test"]
    eval_dataset = eval_dataset.map(tokenize_function, batched=False)


    # Train model
    training_args = TrainingArguments(**trainer_config["training_arguments"])
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
    tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    main()