import csv
from utils import *
import subprocess

from transformers import AutoModel
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer

models = []
# Load model and dataset from csv file
with open("../computer-vision_{}".format(csv_name)) as file:
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
model = None
n_experiment = 1
# Switch for the different kinds of libraries, only transformers is supported for now
if(model_data["library"] == "transformers"):
    model = AutoModel.from_pretrained(model_data["model_name"])

# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    tuning_criterion=tuning_criterion,
)


model_data["model_name"] = model_data["model_name"].replace("/", "-")
# The saving directory will be of the type models/computer-vision/model/id
save_dir = "models/{}/{}".format(model_data["category"], model_data["model_name"])
energy_output_file = "quantization_energy_output_{}_exp{}".format(model_data["model_name"], n_experiment)
print("Start Quantization for model {}".format(model_data["model_name"]))
subprocess.run(["energibridge", "-o", "{}".format(energy_output_file)])
quantizer = INCQuantizer.from_pretrained(model=model)
# The directory where the quantized model will be saved
# Quantize and save the model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
print("End Quantization for model {}, saved to {}".format(model_data["model_name"], save_dir))
