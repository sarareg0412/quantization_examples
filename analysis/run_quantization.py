import evaluate
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from transformers import AutoModel, pipeline
from utils import *

import sys

dataset = None

# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)


def eval_func(model):
    print(f"EVAL_FN, DATASET = NONE IS {dataset is None}")
    pipe = pipeline(model=model)
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    predictions = []
    # Iterate through the test split
    for object in dataset:
        # Load object and label truth label from the dataset
        object = object[dataset.column_names[0]]  # Assume the object column name is the first one
        label = object[dataset.column_names[-1]]  # Assume the label column name is the last one

        # Infer the object label using the model
        prediction = pipe(object)
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        prediction = prediction[0]['label'] if isinstance(prediction, list) else prediction['label']

        # Append ground truth label and predicted label for "accuracy" evaluation
        references.append(label)
        predictions.append(model.config.label2id[prediction])  # Map the predicted label using the model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    exact_match = evaluate.load("exact_match")
    exact_match_score = exact_match.compute(predictions=predictions, references=references)
    return exact_match_score


def run_quantization(save_dir, line):
    model_data = get_model_data_from_line(line)
    # Switch for the different kinds of libraries, only transformers is supported for now
    match model_data["library"]:
        case "transformers":
            model = AutoModel.from_pretrained(model_data["model_name"])
        case "sentence-similarity":
            model = SentenceTransformer(model_data["model_name"])
        case "keras":
            model = from_pretrained_keras(model_data["model_name"])
        case _:
            model = None
    # TODO add specific metric
    quantizer = INCQuantizer.from_pretrained(model=model,
                                             eval_fn=eval_func
                                             )
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)


if __name__ == "__main__":
    dataset = get_dataset_from_name(sys.argv[3], sys.argv[4], QUANT_SPLIT_PERCENT)
    run_quantization(sys.argv[1], sys.argv[2])
