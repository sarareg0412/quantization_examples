import os
from utils import *
from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator

from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras


def run_evaluation_from_line(quantized, line):
    model_data = {csv_header[i]: line[i] for i in range(len(line))}
    data = load_dataset(model_data["dataset_name"], model_data["dataset_config"], split="test").shuffle(seed=42)
    task_evaluator = evaluator(model_data["ask"])

    # If we want to evaluate the NOT quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model = model_data["model_name"]
    else:
        # The quantized model is located in the directory like: ./category/model_name_formatted/config
        quantized_model_path = os.path.join(os.getcwd(), model_data["category"],
                                            format_name(model_data["model_name"]), "config")
        # Switch to retrieve the model class from the different kinds of libraries
        match model_data["model_library"]:
            case "transformers":
                model = AutoModel.from_pretrained(quantized_model_path)
            case "sentence-similarity":
                model = SentenceTransformer(quantized_model_path)
            case "keras":
                model = from_pretrained_keras(quantized_model_path)
            case _:
                model = None

    # Evaluate the model (performs inference too)
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=data
    )
    print(eval_results)
