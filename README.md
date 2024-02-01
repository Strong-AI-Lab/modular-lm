# Independent Causal Language Models

Repository for the ICLM project. This repository contains the code for the fine-tuning a modular ICLM model, as well as the code for the clusters visualisation. The ICLM model inherits from [Huggingfaces' PreTrainedModel](https://huggingface.co/docs/transformers/main_classes/model) so it and its fine-tuned versions can be used as any other model from the [transformers library](https://huggingface.co/docs/transformers/index).


## Installation

To install the repository, use the following command:

```
git clone https://github.com/Strong-AI-Lab/modular-lm.git
```

The data used for the experiments is **not** provided as part of this repository. We use the data from the [Logical-and-abstract-reasoning](https://github.com/Strong-AI-Lab/Logical-and-abstract-reasoning) project. To download the data and use them directly with this repository, use the following command:
```
git clone https://github.com/Strong-AI-Lab/Logical-and-abstract-reasoning.git
```
Makse sure that the two repositories are in the same folder.

To install the dependencies in a virtual environment, use the following:
```
cd modular-lm
python -m venv env/
source env/bin/activate
pip install -r requirements.txt
```



## Fine-Tuning

To fine-tune a model, use the following command:
```
python fine_tuning.py config/model/<model_name>.yaml config/router/<router_name>.yaml config/data/<data_name>.yaml config/trainer/<trainer_name>.yaml
```

By default, the resulting weights are stored in the folder `fine-tuning-saves`.

## Multi-GPU Fine-Tuning

We allow multi-GPU fine-tuning using [deepsepeed](https://github.com/microsoft/DeepSpeed):
```
deepspeed --num_gpus <nb_gpus> fine_tuning.py config/model/<model_name>.yaml config/router/<router_name>.yaml config/data/<data_name>.yaml config/trainer/<trainer_name>.yaml --deepspeed config/trainer/deepspeed_config.json
```



## Evaluation

To evaluate a trained model, use the following command:
```
python run.py config/model/<model_name>.yaml config/data/<data_name>.yaml
```
The model must be a Modular model and can be loaded like a regular transformers model (**not** compatible with `AutoModelForCausalLM`). The evaluation results are saved in the folder `inference-results`. Individual modules can also be used for inference with the flag `--module <module_name>`.



## Clusters Visualisation

The ICLM model uses a routing component to direct the input to the different modules. The clusters visualisation allows to visualise the clusters obtained by routing. To compute the clusters, run the following:
```
python clustering.py config/model/<model_config>>.yaml config/data/<data_config>.yaml --batch_size <batch_size>
```

Once the clusters are computed, the results are saved in the folder `cluster_results`. The centroids obtained with the clustering method are also saved. You can re-run the visualisation without re-computing the clusters by adding the flag `--saved_clusters_path <path_to_clusters>`:
```	
python clustering.py config/model/<model_config>>.yaml config/data/<data_config>.yaml --batch_size <batch_size> --saved_clusters_path <path_to_clusters>
```

You can also visualise the clusters obtained with the router component of a modular model. To do so, you need to add the flag `--router_config <path_to_router_config>`, and optionally the flag `--saved_clusters_path <path_to_clusters>` if you want to use an existing save (most cases):
```
python clustering.py config/model/<model_config>.yaml config/data/<data_config>.yaml --batch_size <batch_size> --router_config config/router/<router_config>.yaml --saved_clusters_path <path_to_clusters>
```

