from functools import partial

from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ViTImageProcessor
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
import evaluate
from optimum.intel.neural_compressor import INCQuantizer

# Load model and dataset
model_name = "nateraw/vit-base-beans"
# Taken from the model's page by clicking the "use in Transformer button"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=3)
extractor = AutoFeatureExtractor.from_pretrained(model_name)  # Extremely important
eval_dataset = load_dataset("beans", split="test")
save_dir = "./beans_exp_static/model_inc"

processor = ViTImageProcessor.from_pretrained(model_name)

def preprocess_function(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = extractor([x for x in example_batch['image']], return_tensors='pt')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


def eval_func(model):
    task_evaluator = evaluate.evaluator("image-classification")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        feature_extractor=extractor,
        data=eval_dataset.with_transform(preprocess_function),
        metric=evaluate.load("accuracy"),
        input_column="image",
        label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]


quantizer = INCQuantizer.from_pretrained(model=model, eval_fn=eval_func)
#quantizer = INCQuantizer.from_pretrained(model=model)

calibration_dataset = quantizer.get_calibration_dataset(
    dataset_name="beans",
    preprocess_function=preprocess_function,
    num_samples=104
)

# Set up quantization configuration
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="static",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)


# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    save_directory=save_dir)