import os
import sys

from utils import *
from transformers import pipeline
from datasets import load_dataset
import evaluate
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

    pipe = pipeline(model_data["task"], model=model, image_processor=processor)
    acc = evaluate.load("accuracy")
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    predictions = []

    # Iterate through the validation set or any other split
    for example in data:
        # Load object and label truth label from the dataset
        object = example[data.column_names[0]]  # Assume the object column name is the first one
        label = example[data.column_names[-1]]  # Assume the label column name is the last one

        # Classify the image using the classification model
        prediction = pipe(object)

        # Assuming 'prediction' contains class probabilities or labels
        predicted_label = prediction[0]['label'] if isinstance(prediction, list) else prediction['label']

        # Append ground truth label and predicted label for accuracy evaluation
        references.append(label)
        predictions.append(predicted_label)

    # Calculate accuracy using the loaded accuracy metric
    accuracy_score = acc.compute(predictions=predictions, references=references)

    print(f"Inference accuracy is : {accuracy_score}")

if __name__ == "__main__":
    run_evaluation_from_line(sys.argv[1], sys.argv[2])

