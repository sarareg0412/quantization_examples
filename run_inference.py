import os

from transformers import pipeline
from datasets import load_dataset
from evaluate import evaluator

from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras


def run_evaluation_from_hub(model_library, quantized, category, model_task, model_name, dataset_name, dataset_config):
    data = load_dataset(dataset_name, dataset_config, split="test").shuffle(seed=42)
    task_evaluator = evaluator(model_task)

    # If we don't want to evaluate the quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model = model_name
    else:
        # The quantized model is located in the directory like: ./models/category/model_name_formatted/config
        quantized_model_path = os.path.join(os.getcwd(), "models", category, model_name.replace("/", "-"), "config")
        match model_library:
            case "transformers":
                model = AutoModel.from_pretrained(quantized_model_path)
            case "sentence-similarity":
                model = SentenceTransformer(quantized_model_path)
            case "keras":
                model = from_pretrained_keras(quantized_model_path)
            case _:
                model = None

    # 1. Pass a model name or path
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=data
    )