from functools import partial

from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from evaluate import load
from optimum.intel.neural_compressor import INCQuantizer
from utils import get_processor_from_category
# Load model and dataset
model_name = "nateraw/vit-base-beans"
# Taken from the model's page by clicking the "use in Transformer button"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=3)
processor = AutoFeatureExtractor.from_pretrained(model_name)  # Extremely important
dataset = load_dataset("beans", split="test").train_test_split(train_size=0.5, seed=42)["train"]  # Use x% of test dataset

def eval_func(model):
    # TODO add tokenizer
    pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    predictions = []
    # Iterate through the test split
    for data in dataset:
        # Load object and label truth label from the dataset
        object = data[dataset.column_names[1]]  # Assume the object column name is the first one
        label = data[dataset.column_names[-1]]  # Assume the label column name is the last one

        # Infer the object label using the model
        prediction = pipe(object)
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        prediction = prediction[0]['label'] if isinstance(prediction, list) else prediction['label']

        # Append ground truth label and predicted label for "accuracy" evaluation
        references.append(str(label))
        predictions.append(model.config.label2id[prediction])  # Map the predicted label using the model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    exact_match = load("exact_match")
    exact_match_score = exact_match.compute(predictions=predictions, references=references)
    return exact_match_score["exact_match"]


# Set up quantization configuration
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)

save_dir = "./computer-vision/nateraw-vit-base-beans/DUMP_config"

quantizer = INCQuantizer.from_pretrained(model=model, eval_fn=eval_func)

# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)