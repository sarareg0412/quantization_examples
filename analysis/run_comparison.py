
import evaluate
from utils import *
from transformers import pipeline
from datasets import load_dataset

exact_match = evaluate.load("exact_match", module_type="comparison")

def run_comparison(line):
    # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
    model_data_names = csv_header
    model_data_names.append("full_line")
    line = line.split(",")
    model_data = {model_data_names[i]: line[i] for i in range(len(line))}
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = data.train_test_split(train_size=0.5, seed=SEED)["test"]  # Use 50% of test dataset to run comparison

    # Get processor (image processor, tokenizer etc.)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    # Retrieve quantized model
    q_model = get_model_from_library(model_data["library"], model_data["task"],
                                     get_quantized_model_path(model_data["category"], model_data["model_name"]))
    nq_pipe = pipeline(model_data["task"], model=model_data["model_name"], image_processor=processor)
    # Quantized model pipeline for inference
    q_pipe = pipeline(model_data["task"], model=q_model, image_processor=processor)
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    nq_predictions = []
    q_predictions = []

    # Iterate through the validation set or any other split
    for example in data:
        # Load object and label truth label from the dataset
        object = example[data.column_names[0]]  # Assume the object column name is the first one
        label = example[data.column_names[-1]]  # Assume the label column name is the last one

        # Infer the object label using the model
        nq_prediction = nq_pipe(object)

        # Since there might be multiple labels with multiple scores associated, we get the first one.
        predicted_label = prediction[0]['label'] if isinstance(prediction, list) \
            else prediction['label']

        # Append ground truth label and predicted label for accuracy evaluation
        references.append(label)
        predictions.append(
            model.config.label2id[predicted_label])  # Map the predicted label using the model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    accuracy_score = exact_match.compute(predictions1=nq_predictions, predictions2=q_predictions, references=references)

    print(f"Inference accuracy is : {accuracy_score}")