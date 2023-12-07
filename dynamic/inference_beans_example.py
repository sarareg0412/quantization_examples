
from datasets import load_dataset
from transformers import AutoModel
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
import evaluate
from optimum.intel.neural_compressor import INCQuantizer

# Load model and dataset
model_name = "nateraw/vit-base-beans"
model = AutoModel.from_pretrained(model_name)
eval_dataset = load_dataset("beans", split="test")
#extractor = AutoFeatureExtractor.from_pretrained(model_name)  # Extremely important
"""
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
 """
# Set up quantization configuration
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)

save_dir = "./beans_dynamic_model"

quantizer = INCQuantizer.from_pretrained(model=model)

# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
