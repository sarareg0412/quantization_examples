import os
import sys

from utils import *
from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator

from transformers import AutoImageProcessor
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras


def run_evaluation_from_line(quantized, line):
    # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
    model_data_names = csv_header
    model_data_names.append("full_line")
    line = line.split(",")
    model_data = {model_data_names[i]: line[i] for i in range(len(line))}
    data = load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test").shuffle(seed=42)
    task_evaluator = evaluator(model_data["task"])

    quantized = True if (quantized == "True") else False
    # The quantized model is located in the directory like: ./category/model_name_formatted/config
    model_path = os.path.join(os.getcwd(), model_data["category"],
                                        format_name(model_data["model_name"]), "config")
    # If we want to evaluate the NOT quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model_path = model_data["model_name"]

    # Switch to retrieve the model class from the different kinds of libraries
    match model_data["library"]:
        case "transformers":
            model = get_model_from_task(model_data["task"], model_path)
        case "sentence-similarity":
            model = SentenceTransformer(model_path)
        case "keras":
            model = from_pretrained_keras(model_path)
        case _:
            model = None

    match model_data["category"]:
        case "computer-vision":
            processor = AutoImageProcessor.from_pretrained(model_data["model_name"])
        case _:
            processor = None

    # Evaluate the model (performs inference too)
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=data,
        label_column=data.column_names[-1],  # We assume the last column is the labels one
        feature_extractor=processor,
        tokenizer=processor,
        label_mapping=model.config.label2id     # doesn't work without it, supposedly it's different per each dataset
    )
    print(eval_results)


if __name__ == "__main__":
    run_evaluation_from_line(sys.argv[1], sys.argv[2])

