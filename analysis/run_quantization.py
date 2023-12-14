from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import from_pretrained_keras

import sys


# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    tuning_criterion=tuning_criterion,
)


def run_quantization(save_dir, model_library, model_name):
    # Switch for the different kinds of libraries, only transformers is supported for now
    match model_library:
        case "transformers":
            model = AutoModel.from_pretrained(model_name)
        case "sentence-similarity":
            model = SentenceTransformer(model_name)
        case "keras":
            model = from_pretrained_keras(model_name)
        case _:
            model = None
    quantizer = INCQuantizer.from_pretrained(model=model)
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)


if __name__ == "__main__":
    run_quantization(sys.argv[1], sys.argv[2], sys.argv[3])
