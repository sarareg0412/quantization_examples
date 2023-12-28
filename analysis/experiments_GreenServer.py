import csv
import os

from utils import *
import subprocess
from run_comparison import run_comparison
from run_quantization import run_quantization

DEBUG = False

def get_models_line_from_csv(category):
    filename = "../models_csv/{}".format(csv_name)
    if category is not None:
        filename = "../models_csv/{}_{}".format(category, csv_name)
    with open(filename) as file:
        csvReader = csv.reader(file)
        # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
        csvReader.__next__()        # Skip header
        # choose the first 5 lines out of every 50 lines of the file
        lines = [",".join(line) for i, line in enumerate(csvReader) if i % 50 <5]

    return lines


def quantize_and_measure_consumption():
    # Load models from csv file
    top_N_models = get_models_line_from_csv("computer-vision")

    line = top_N_models[3]    # beans model
    model_data = get_model_data_from_line(line)

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
        if DEBUG:
            run_quantization(energy_output_file, line)
        else:
            subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file),
                            "python", "run_quantization.py", "{}".format(save_model_dir),
                                                             "{}".format(line),
                                                             "{}".format(model_data["dataset"]),
                                                             "{}".format(model_data["dataset_config_name"]),
                            ])
        print("END QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))


def infer_and_measure_consumption(quantized):
    # Load models from csv file
    top_N_models = get_models_line_from_csv("computer-vision")
    line = top_N_models[3]    # beans model

    model_data = get_model_data_from_line(line)
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
        print("START INFERENCE FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                                model_data["model_name"], n_experiment))
        subprocess.run(["../energibridge", "-o", "{}".format(energy_output_file),
                        "python", "run_inference.py", "{}".format(str(quantized)),
                        "{}".format(line)])
        print("END INFERENCE FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                              model_data["model_name"], n_experiment))


def compare_models():
    # Load models from csv file
    top_N_models = get_models_line_from_csv("computer-vision")
    line = top_N_models[3]  # beans model
    model_data = get_model_data_from_line(line)
    #for n_experiment in range(0, N_EXPERIMENTS + 1):
    run_comparison(model_data)



quantize_and_measure_consumption()
#infer_and_measure_consumption(True)
#compare_models()