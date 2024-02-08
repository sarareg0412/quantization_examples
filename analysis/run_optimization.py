from evaluate import load, evaluator
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from utils import *
import sys

dataset = None
processor = None
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
    # SequenceClassification is actually "text-classification"
    task_evaluator = evaluator(model_data['task'])
    results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=processor,
        data=dataset,
        metric=load("accuracy"),
        # input_column=dataset.column_names[0],
        # label_column="labels",
        # label_mapping=model.config.label2id,
    )
    return results["accuracy"]


def run_optimization(save_dir):
    model = get_model_from_library(model_data["library"], model_data["category"], model_data["model_name"])

    quantizer = INCQuantizer.from_pretrained(model=model,
                                             eval_fn=eval_func
                                             )
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)


if __name__ == "__main__":
    model_data = get_model_data_from_line(sys.argv[1])
    dataset = get_split_dataset(model_data, train_size=0.5, seed=SEED, split='test')
    processor = AutoTokenizer.from_pretrained(model_data['model_name'], model_max_length=512)
    #processor = get_processor_from_category(model_data[̈́'category'], model_data[̈́'model_name'])
    run_optimization(sys.argv[2])
