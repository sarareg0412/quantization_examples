import csv
import os

from utils import csv_name, get_model_data_from_line, format_name, N_EXPERIMENTS
import subprocess
from run_comparison import run_comparison
from run_experiments import run_experiments

cat = "SequenceClassification"
model_id = 1


def get_models_line_from_csv(category):
    filename = "./INC_models/INCModelFor{}_{}".format(category, csv_name)
    with open(filename) as file:
        csvReader = csv.reader(file)
        # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
        csvReader.__next__()  # Skip header
        # choose the first 5 lines out of every 50 lines of the file
        lines = [",".join(line) for i, line in enumerate(csvReader) if i % 50 < 5]

    return lines


def quantize_and_measure_consumption():
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]  # bart-large-cnn
    model_data = get_model_data_from_line(line)

    model_name_formatted = format_name(model_data["model_name"])

    # The saving directory of the model weights will follow the naming convention like
    # ./INCModelFor.../model_name_formatted/config
    save_model_dir = "{}/{}/config".format(model_data["category"], model_name_formatted)
    # The model's energy data files will be csv and in the directory following the naming convention like
    # ./computer-vision/model_name_formatted/quant_energy_data
    save_energy_file_dir = "{}/{}/quant_energy_data".format(model_data["category"], model_name_formatted)
    # Preliminary creation of the needed directory to save the output file, or the energibridge command won't work
    os.makedirs(save_energy_file_dir, exist_ok=True)
    for n_experiment in range(0, N_EXPERIMENTS + 1):
        # The output file will be named model-name-formatted_quant_exp0.csv
        energy_output_file = "{}/{}_quant_exp{}.csv".format(save_energy_file_dir, model_name_formatted,
                                                            f"0{n_experiment}" if n_experiment in range(0, 10)
                                                            else n_experiment)
        print("START QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))
        subprocess.run([
            # "../energibridge", "-o", "{}".format(energy_output_file),
            "optimum-cli", "inc", "quantize", "--model", "{}".format(model_data["model_name"]),
            "--output", "{}".format(save_model_dir)
        ])
        print("END QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))


def infer_and_measure_consumption(quantized):
    # Load models from csv file
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]  # cardiffNLP

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
        # add
        # time.sleep(5)   # Sleep for 5 seconds
        # The output file will be named model-name-formatted_Q_inf_exp0.csv
        energy_output_file = "{}/{}_{}inf_exp{}.csv".format(save_energy_file_dir,
                                                            model_name_formatted,
                                                            "Q_" if quantized else "",
                                                            f"0{n_experiment}" if n_experiment in range(0, 10)
                                                            else n_experiment)
        print("START INFERENCE FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                               model_data["model_name"], n_experiment))
        subprocess.run([
                        # "../energibridge", "-o", "{}".format(energy_output_file),
                        "python", "run_inference.py", "{}".format(str(quantized)),
                        "{}".format(line)])
        print("END INFERENCE FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                             model_data["model_name"], n_experiment))


def compare_models():
    # Load models from csv file
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]
    # for n_experiment in range(0, N_EXPERIMENTS + 1):
    run_comparison(line)


def use_evaluate_hf():
    # Load models from csv file
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]  # cardiffNLP
    subprocess.run(["python", "evaluate_HF.py", "{}".format(line), 'seqeval'])


def run_experiments_and_create_csv():
    # Load models from csv file
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]
    run_experiments(line)


def run_optimization():
    # Load models from csv file
    top_N_models = get_models_line_from_csv(cat)
    line = top_N_models[model_id]  # cardiffNLP
    model_data = get_model_data_from_line(line)
    save_model_dir = "{}/{}/opt_quant/config".format(model_data["category"], format_name(model_data['model_name']))
    print("START QUANTIZATION OPTIMIZATION FOR MODEL {}".format(model_data["model_name"]))
    subprocess.run(["python", "run_optimization.py", "{}".format(line), "{}".format(save_model_dir)])

#quantize_and_measure_consumption()
#infer_and_measure_consumption(True)
#infer_and_measure_consumption(False)
#compare_models()
#use_evaluate_hf()
#run_experiments_and_create_csv()
run_optimization()