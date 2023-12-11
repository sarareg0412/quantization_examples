import csv
from utils import *
import subprocess



models = []
# Load model and dataset from csv file

with open("../models_csv/computer-vision_{}".format(csv_name)) as file:
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
for model_data in top_N_models:
    n_experiment = 1
    model_name_formatted = model_data["model_name"].replace("/", "-")

    # The saving directory will be of the type models/computer-vision/model_name_formatted
    save_dir = "models/{}/{}".format(model_data["category"], model_name_formatted)
    energy_output_file = "{}/energy_output_exp{}.csv".format(save_dir, n_experiment)
    print("Start Quantization for model {}".format(model_data["model_name"]))
    subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file),
                    "python", "run_energy.py", "{}".format(save_dir), "{}".format(model_data["library"]), "{}".format(model_data["model_name"])])
    print("End Quantization for model {}, saved to {}".format(model_data["model_name"], save_dir))
    n_experiment += 1
