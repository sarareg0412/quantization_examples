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
            data = {header[i]: line[i] for i in range(len(line))}
            # The model also has the "full_line" attribute which is the line read from the csv file
            data["full_line"] = ','.join(line)
            models.append(data)

    return models


def quantize_and_measure_consumption():
    # Load models from csv file
    top_N_models = get_models_from_csv("computer-vision")

    model_data = top_N_models[3]    # beans model

    model_name_formatted = format_name(model_data["model_name"])

    # The saving directory of the model weights will follow the naming convention like
    # ./computer-vision/model_name_formatted/config
    save_model_dir = "{}/{}/config".format(model_data["category"], model_name_formatted)
    # The model's energy data files will be csv and in the directory following the naming convention like
    # ./computer-vision/model_name_formatted/quant_energy_data
    save_energy_file_dir = "{}/{}/quant_energy_data".format(model_data["category"], model_name_formatted)
    # Preliminary creation of the needed directory to save the output file, or the energibridge command won't work
    os.makedirs(save_energy_file_dir, exist_ok=True)
    for n_experiment in range(0, N_EXPERIMENTS + 1):
        # The output file will be named model-name-formatted_quant_exp0.csv
        energy_output_file = "{}/{}_quant_exp{}.csv".format(save_energy_file_dir, model_name_formatted, n_experiment)
        print("START QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))
        subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file),
                        "python", "run_quantization.py", "{}".format(save_model_dir),
                                                         "{}".format(model_data["library"]),
                                                         "{}".format(model_data["model_name"])])
        print("END QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))


def evaluate_and_measure_consumption(quantized):
    # Load models from csv file
    top_N_models = get_models_from_csv("computer-vision")
    model_data = top_N_models[3]    # beans model
    model_name_formatted = format_name(model_data["model_name"])
    # The model's energy data files will be csv and in the directory following the naming convention like
    # ./computer-vision/model_name_formatted/inf_energy_data/quant or non_quant based on the quantized parameter
    save_energy_file_dir = "{}/{}/inf_energy_data/{}".format(model_data["category"],
                                                             model_name_formatted,
                                                             "quant" if quantized else "non_quant")
    # Preliminary creation of the needed directory to save the output file, or the energibridge command won't work
    os.makedirs(save_energy_file_dir, exist_ok=True)
    for n_experiment in range(0, N_EXPERIMENTS + 1):
        # The output file will be named model-name-formatted_Q_inf_exp0.csv
        energy_output_file = "{}/{}_{}inf_exp{}.csv".format(save_energy_file_dir,
                                                            model_name_formatted,
                                                            "Q_" if quantized else "",
                                                            n_experiment)
        print("START EVALUATION FOR {} MODEL {} - EXP {}".format("QUANTIZED" if quantized else "",
                                                                 model_data["model_name"], n_experiment))
        subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file),
                        "python", "run_inference.py", "{}".format(str(quantized)),
                                                      "{}".format(model_data["full_line"])])
        print("END EVALUATION FOR {} MODEL {} - EXP {}".format("QUANTIZED" if quantized else "",
                                                               model_data["model_name"], n_experiment))


#quantize_and_measure_consumption()
evaluate_and_measure_consumption(True)