from evaluate import load, evaluator
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel import INCQuantizer
from transformers import pipeline
from utils import *
import sys

dataset = None
processor = None
model = None
# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)

# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)


def eval_func(model_eval):
    pipe.model = model_eval
    """
        Currently accepted tasks are:
        - `"image-classification"`
        - `"question-answering"`
        - `"text-classification"`
        - `"token-classification"`
    """
    task_evaluator = evaluator(model_data['task'])

    args = {
        "model_or_pipeline": pipe,
        "data": dataset,
        "metric": get_metric_from_category(model_data['category'])
    }

    if model_data['category'] == 'INCModelForSequenceClassification':
        args["label_mapping"]=model_eval.config.label2id

    results = task_evaluator.compute(
        **args
    )

    return list(results.values())[0]


def run_optimization(save_dir):

    quantizer = INCQuantizer.from_pretrained(model=model,
                                             eval_fn=eval_func
                                             )
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)


if __name__ == "__main__":
    model_data = get_model_data_from_line(sys.argv[1])
    dataset = get_split_dataset(model_data, train_size=0.5, seed=SEED, split='test')
    processor = get_processor_from_category(model_data['category'], model_data['model_name'])
    #processor = AutoTokenizer.from_pretrained(model_data['model_name'], model_max_length=512)
    model = get_model_from_library(model_data["library"], model_data["category"], model_data["model_name"])
    pipe = pipeline(model_data['task'], model=model, tokenizer=processor)

    run_optimization(sys.argv[2])

