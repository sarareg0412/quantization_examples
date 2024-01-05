from evaluate import load, evaluator
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from transformers import AutoModelForImageClassification, pipeline
from utils import *
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
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

d_q_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

def eval_func(model):
    # pipe = pipeline(
    #    self.task,
    #    model=model_or_pipeline,
    #    tokenizer=tokenizer,
    #    feature_extractor=feature_extractor,
    #    device=device,
    # )
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    # TODO add tokenizer
    pipe = pipeline(model_data["task"], model=model, feature_extractor=processor)
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


def eval_func2(model):
    task_evaluator = evaluator("image-classification")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        feature_extractor=get_processor_from_category(model_data["category"], model_data["model_name"]),
        data=dataset,
        metric=load("accuracy"),
        input_column=dataset.column_names[0],
        label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]


def run_quantization(save_dir):
    # Switch for the different kinds of libraries, only transformers is supported for now
    model = get_ORT_model_from_library(model_data["library"], model_data["task"], model_data["model_name"])
    processor = get_extractor_from_category(model_data["category"], model_data["model_name"])
    quantizer = ORTQuantizer.from_pretrained(model)
    # skipping saving the onnx checkpoint and tokenizer
    #model.save_pretrained(onnx_path)
    #processor.save_pretrained(onnx_path)
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=d_q_config, save_dir=save_dir)


if __name__ == "__main__":
    model_data = get_model_data_from_line(sys.argv[2])
    dataset = get_dataset_from_name(model_data["dataset"], model_data["dataset_config_name"], QUANT_SPLIT_PERCENT)
    run_quantization(sys.argv[1])
