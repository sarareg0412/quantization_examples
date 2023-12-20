
import evaluate
from utils import *
from transformers import pipeline
from datasets import load_dataset

exact_match = evaluate.load("exact_match", module_type="comparison")

def run_comparison(model_data):
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = data.train_test_split(train_size=0.5, seed=SEED)["test"]  # Use 50% of test dataset to run comparison

    # Get processor (image processor, tokenizer etc.)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    # No need to retrieve the non quantized model as we only need its name to retrieve it from the hub
    # Retrieve quantized model by its configuration.
    q_model = get_model_from_library(model_data["library"], model_data["task"],
                                     get_quantized_model_path(model_data["category"], model_data["model_name"]))
    # Setup non quantized and quantized model pipeline for inference
    nq_pipe = pipeline(model_data["task"], model=model_data["model_name"], image_processor=processor)
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
        q_prediction = q_pipe(object)
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        nq_label = nq_prediction[0]['label'] if isinstance(nq_prediction, list) \
            else nq_prediction['label']
        q_label = q_prediction[0]['label'] if isinstance(q_prediction, list) \
            else q_prediction['label']

        # Append ground truth label and predicted label for accuracy evaluation
        references.append(label)
        nq_predictions.append(q_model.config.label2id[nq_label])  # Map the NQ predicted label using the q model's label2id attribute
        q_predictions.append(q_model.config.label2id[q_label])    # Map the Q predicted label using the q model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    exact_match_score = exact_match.compute(predictions1=nq_predictions, predictions2=q_predictions, references=references)

    print(f"Exact match score is : {exact_match_score}")