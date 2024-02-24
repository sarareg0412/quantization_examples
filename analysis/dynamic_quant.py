import time

from run_comparison import run_comparison
from run_experiments import run_experiments
from utils import *
import subprocess

top_5_filename = f"./INC_models/{csv_name}"


def get_models_line_from_csv(category):
    filename = "./INC_models/INCModelFor{}_{}".format(category, csv_name)
    with open(filename) as file:
        csvReader = csv.reader(file)
        # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
        csvReader.__next__()  # Skip header
        # choose the first 5 lines out of every 50 lines of the file
        lines = [",".join(line) for i, line in enumerate(csvReader) if i % 50 < 5]

    return lines


def get_top_5_models_line_from_csv():
    with open(top_5_filename) as file:
        csvReader = csv.reader(file)
        # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
        csvReader.__next__()  # Skip header
        # choose the first 5 lines out of every 50 lines of the file
        lines = [",".join(line) for line in csvReader]

    return lines


def quantize_and_measure_consumption():
    # Load models from csv file
    top_N_models = get_top_5_models_line_from_csv()

    # Quantize the N_MODELS models of each category:
    model_data = get_model_data_from_line(top_N_models[0])
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
        time.sleep(5)  # Sleep for 5 seconds
        # The output file will be named model-name-formatted_quant_exp0.csv
        energy_output_file = "{}/{}_quant_exp{}.csv".format(save_energy_file_dir, model_name_formatted,
                                                            f"0{n_experiment}" if n_experiment in range(0, 10)
                                                            else n_experiment)
        print("START QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))
        subprocess.run([
            "/home/tdurieux/git/EnergiBridge/target/release/energibridge", "-o", "{}".format(energy_output_file),
            "optimum-cli", "inc", "quantize", "--model", "{}".format(model_data["model_name"]),
            "--output", "{}".format(save_model_dir)
            # "python", "run_optimization.py", "{}".format(save_model_dir),
            #                                 "{}".format(line)
        ])
        print("END QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))


def infer_and_measure_consumption(quantized):
    # Load models from csv file
    top_N_models = get_top_5_models_line_from_csv()
    # Evaluate the N_MODELS models of each category:
    line = top_N_models[1]  # QuestionAnswering
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
        time.sleep(5)  # Sleep for 5 seconds
        # The output file will be named model-name-formatted_Q_inf_exp0.csv
        energy_output_file = "{}/{}_{}inf_exp{}.csv".format(save_energy_file_dir,
                                                            model_name_formatted,
                                                            "Q_" if quantized else "",
                                                            f"0{n_experiment}" if n_experiment in range(0, 10)
                                                            else n_experiment)
        print("START EVALUATION FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                                model_data["model_name"], n_experiment))
        subprocess.run(
            ["/home/tdurieux/git/EnergiBridge/target/release/energibridge", "-o", "{}".format(energy_output_file),
             "python", "run_inference.py", "{}".format(str(quantized)),
             "{}".format(line)])
        print("END EVALUATION FOR {}MODEL {} - EXP {}".format("QUANTIZED " if quantized else "",
                                                              model_data["model_name"], n_experiment))


def run_optimization():
    # Load models from csv file
    top_N_models = get_top_5_models_line_from_csv()
    # For now, the optimization process will be done only for the QuestionAnswering task
    line = top_N_models[1]  # Question Answering
    model_data = get_model_data_from_line(line)
    model_name_formatted = format_name(model_data["model_name"])
    save_model_dir = "{}/{}/optim/config".format(model_data["category"], model_name_formatted)
    save_energy_file_dir = "{}/{}/optim/quant_energy_data".format(model_data["category"], model_name_formatted)
    os.makedirs("{}/{}/optim", exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_energy_file_dir, exist_ok=True)
    for n_experiment in range(0, N_EXPERIMENTS + 1):
        time.sleep(5)  # Sleep for 5 seconds
        # The output file will be named model-name-formatted_quant_exp0.csv
        energy_output_file = "{}/{}_quant_exp{}.csv".format(save_energy_file_dir, model_name_formatted,
                                                            f"0{n_experiment}" if n_experiment in range(0, 10)
                                                            else n_experiment)
        print("START QUANTIZATION OPTIMIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))
        subprocess.run(
            ["/home/tdurieux/git/EnergiBridge/target/release/energibridge", "-o", "{}".format(energy_output_file),
             "python", "run_optimization.py", "{}".format(line),
             "{}".format(save_model_dir)])
        print("END QUANTIZATION FOR MODEL {} - EXP {}".format(model_data["model_name"], n_experiment))


def compare_models():
    # Load models from csv file
    top_N_models = get_top_5_models_line_from_csv()
    line = top_N_models[2]  # cardiffNLP
    run_comparison(line)


def run_validation():
    for model_line in get_top_5_models_line_from_csv():
        run_experiments(model_line)
    # TODO run again with USE_OPTIM = True
    #optim_model = get_top_5_models_line_from_csv()[1]
    #run_experiments(optim_model)


#quantize_and_measure_consumption()
run_optimization()
infer_and_measure_consumption(True)
infer_and_measure_consumption(False)
#run_validation()
