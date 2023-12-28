import evaluate
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras
from evaluation_functions import eval_func
from utils import *

import sys


# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)


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

    dataloader = get_dataloader_from_dataset_name(model_data["dataset"], model_data["dataset_config_name"], QUANT_SPLIT_PERCENT)
    # TODO add specific metric
    quantizer = INCQuantizer.from_pretrained(model=model
                                             #eval_fn=eval_func(model=model, dataset=dataloader, metric=evaluate.load("accuracy")))
                                             )
    calib_dataset = quantizer.get_calibration_dataset(dataset_name=model_data["dataset"],
                                                      dataset_config_name=model_data["dataset_config_name"],
                                                      dataset_split="test"
                                                      )
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir, calibration_dataset=calib_dataset)


if __name__ == "__main__":
    run_quantization(sys.argv[1], sys.argv[2])
