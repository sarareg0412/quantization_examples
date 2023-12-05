import csv
from utils import *

from transformers import AutoModel
from datasets import load_dataset

from datasets import load_dataset
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer

models = []
# Load model and dataset from csv file
with open("../{}".format(csv_name)) as file:
    csvReader = csv.reader(file)
    # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
    header = csvReader.__next__()
    for line in csvReader:
        models.append({header[i]: line[i] for i in range(len(line))})


# Get the first 5 elements, every 50 elements
top_N_models = []
for i in range(0, len(models), 50):
    top_N_models.extend(models[i:i + N_MODELS])


# Quantize only the N_MODELS models of each category:
#for model in top_N_models:
model_data = top_N_models[0]
if(model_data["library"] == "transformers"):
    model = AutoModel.from_pretrained(model_data["model_name"])
    eval_dataset = load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test").select(range(300))

# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    tuning_criterion=tuning_criterion,
)
print("Start Quantization")
# The saving directory will be of the type
save_dir = "models/{}/{}".format(model["category"], model["id"])

quantizer = INCQuantizer.from_pretrained(model=model)
# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)