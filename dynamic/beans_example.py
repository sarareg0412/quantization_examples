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

def eval_func(model):
    task_evaluator = evaluate.evaluator("image-classification")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        feature_extractor=extractor,
        data=eval_dataset,
        metric=evaluate.load("accuracy"),
        input_column="image",
        label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]

# Set up quantization configuration
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)

save_dir = "../static/beans_exp_static/model_inc"

quantizer = INCQuantizer.from_pretrained(model=model, eval_fn=eval_func)

# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)