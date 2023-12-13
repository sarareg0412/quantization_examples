import csv
import os

from utils import *
import subprocess


def get_models_from_csv(category):
    models = []
    filename = "../models_csv/{}".format(csv_name)
    if category is not None:
        filename = "../models_csv/{}_{}".format(category, csv_name)
    with open(filename) as file:
        csvReader = csv.reader(file)
        # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
        header = csvReader.__next__()
        # choose the first 5 lines out of every 50 lines of the file
        lines = [line for i, line in enumerate(csvReader) if i % 50 <5]

        for line in lines:
            models.append({header[i]: line[i] for i in range(len(line))})

    return models


def quantize_and_measure_consumption():
    # Load models from csv file
    top_N_models_of_category = get_models_from_csv(None)

    # Quantize the N_MODELS models of each category:
    for model_data in top_N_models_of_category:
        model_name_formatted = model_data["model_name"].replace("/", "-")

        # The saving directory of the model weights will be of the type models/computer-vision/model_name_formatted/
        save_energy_file_dir = "models/{}/{}/energy_data".format(model_data["category"], model_name_formatted)
        save_model_dir = "models/{}/{}/config".format(model_data["category"], model_name_formatted)
        # Preliminary creation of the needed directory to save the output file, or the energibridge command won't work
        os.makedirs(save_energy_file_dir, exist_ok=True)
        for n_experiment in range(0, N_EXPERIMENTS):
            energy_output_file = "{}/energy_output_exp{}".format(save_energy_file_dir, n_experiment)
            print("START QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))
            subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file), "-s", ";"
                            "python", "run_quantization.py", "{}".format(save_model_dir), "{}".format(model_data["library"]),
                                                             "{}".format(model_data["model_name"])])
            print("END QUANTIZATION FOR MODEL {}".format(model_data["model_name"]))


def evaluate_and_measure_consumption(quantized):
    # Load models from csv file
    top_N_models_of_category = get_models_from_csv(None)
    # Evaluate the N_MODELS models of each category:
    for model_data in top_N_models_of_category:
        # Evaluate the quantized version
        if quantized:
            model_name_formatted = model_data["model_name"].replace("/", "-")



#quantize_and_measure_consumption()