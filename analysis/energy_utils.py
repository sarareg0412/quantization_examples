import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns
sns.set()
import numpy as np


def generate_metric_charts(path: str, model_name):
    fig, ax = plt.subplots(figsize=[5, 3])
    ax.set_ylim(0, 60)
    ax.set_xlim(0, 50)

    all_data = []
    if not os.path.isdir(path):
        print(f"Path {path} not found")
    else:
        sorted_files = os.listdir(path)
        sorted_files = sorted(sorted_files, key=lambda x: x.split('_')[2])
        for csv_file in sorted_files:
            if not csv_file.endswith(".csv"):
                continue
            print(f"Reading csv file {csv_file}")
            df = pd.read_csv(os.path.join(path, csv_file))
            print("Done.")
            key = "PACAKGE_ENERGY (W)"
            if "CPU_ENERGY (J)" in df.columns:
                key = "CPU_ENERGY (J)"
            if "PACAKGE0_ENERGY (W)" in df.columns:
                key = "PACAKGE0_ENERGY (W)"
            if "SYSTEM_POWER (Watts)" in df.columns:
                key = "SYSTEM_POWER (Watts)"
            data = df[key].copy().to_list()
            if key != "CPU_POWER (Watts)" and key != "SYSTEM_POWER (Watts)":
                df[key + "_original"] = df[key].copy()
                for i in range(0, len(data)):
                    if i in df[key + "_original"] and i - 1 in df[key + "_original"]:
                        # diff with previous value and convert to watts
                        data[i] = (data[i] - df[key + "_original"][i - 1]) * (1000 / df["Delta"][i])
                    else:
                        data[i] = 0
            data = data[1:-1]  # take out first read
            for i in range(0, len(data)):
                all_data.append({"Time": i, "CPU_POWER (Watts)": data[i]})

        plot = sns.lineplot(data=pd.DataFrame(all_data), x="Time", y="CPU_POWER (Watts)", estimator=np.median,
                            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)), ax=ax, legend=True)
        plot.set(xlabel="Time", ylabel="CPU_POWER (Watts)")
        title = f"{model_name}_quant_energy_data_plot.pdf"
        plot.set_title(title)
        plot.get_figure().savefig(os.path.join(f"./plots/{title}"))
    # plt.show()


def generate_violin_charts(path: str):
    #fig, axes = plt.subplots(figsize=[5, 3])
    plt.figure(figsize=(7,10))
    if not os.path.isdir(path):
        print(f"Path {path} not found")
    else:
        # Path is of the form 'INCModelFor.../model-name-formatted'
        model_name = path.split('/')[-1]
        data_quant = get_data_from_path(os.path.join(path, 'quant_energy_data'))
        data_inf_Q = get_data_from_path(os.path.join(path, 'inf_energy_data/quant'))
        # Sum the quantization energy consumption to the inference of the quantized model
        tot_quant_data = [sum(x) for x in zip(data_quant, data_inf_Q)]
        data_inf_NQ = get_data_from_path(os.path.join(path, 'inf_energy_data/non_quant'))

        df = pd.DataFrame({'Quantized model': tot_quant_data, 'Non Quantized model':data_inf_NQ})

        sns.violinplot(data= df, palette=['tab:blue', 'tab:orange'],  density_norm='width')
        avg_NQ = (sum(data_inf_NQ) / len(data_inf_Q))
        avg_Q = (sum(tot_quant_data) / len(tot_quant_data))
        plt.text(0.75, avg_NQ/4, f'Average energy Q = {avg_Q:.3f} J'
                                      f'\nAverage energy NQ = {avg_NQ:.3f} J\n'
                                      f'Amount of energy saved: {(1-(avg_Q/avg_NQ))*100:.2f}%',
                    fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5))
        # Set labels and title
        plt.ylabel('Energy')
        plt.ylim(0)
        plt.title(f"Model {model_name} Energy Data Plot")
        plt.savefig(os.path.join(f"./plots/{model_name}_energy_data_plot.png"))
    # plt.show()

def get_data_from_path(path):
    all_data = []
    files = os.listdir(path)
    sorted_files = sorted(files, key=lambda x: x.split('_')[-1])
    for csv_file in sorted_files[1:21]:   # skip the first experiment
        if not csv_file.endswith(".csv"):
            continue
        print(f"Reading csv file {csv_file}")
        df = pd.read_csv(os.path.join(path, csv_file))
        key = "PACAKGE_ENERGY (W)"
        if "CPU_ENERGY (J)" in df.columns:
            key = "CPU_ENERGY (J)"
        if "PACAKGE0_ENERGY (W)" in df.columns:
            key = "PACAKGE0_ENERGY (W)"
        if "SYSTEM_POWER (Watts)" in df.columns:
            key = "SYSTEM_POWER (Watts)"
        data = df[key].copy().to_list()
        all_data.append(data[-1] - data[0])  # Here we get the total consumption of 1 experiment

    return all_data

def avg_metric(df: pd.DataFrame, metric_name: str):
    all_data = None
    nb_point = 0
    for metric in df.columns[1:]:
        if metric_name in metric:
            nb_point += 1
            if all_data is None:
                all_data = df[metric].copy()
            else:
                all_data += df[metric]
    return all_data / nb_point


def generate_metric_charts_csv(csv_file):
    all_data = []
    if not os.path.exists(csv_file):
        raise ValueError(f'{csv_file} does not exist')
    df = pd.read_csv(csv_file)
    key = "PACKAGE_ENERGY (W)"
    if "CPU_ENERGY (J)" in df.columns:
        key = "CPU_ENERGY (J)"
    if "PACKAGE_ENERGY (J)" in df.columns:
        key = "PACKAGE_ENERGY (J)"
    if "SYSTEM_POWER (Watts)" in df.columns:
        key = "SYSTEM_POWER (Watts)"
    data = df[key].copy().to_list()
    if key != "CPU_POWER (Watts)" and key != "SYSTEM_POWER (Watts)":
        df[key + "_original"] = df[key].copy()
        for i in range(0, len(data)):
            if i in df[key + "_original"] and i - 1 in df[key + "_original"]:
                # diff with previous value and convert to watts
                data[i] = (data[i] - df[key + "_original"][i - 1]) * (1000 / df["Delta"][i])
            else:
                data[i] = 0
    # data = data[1:-1]
    for i in range(0, len(data)):
        all_data.append({"Time": i, "CPU_POWER (Watts)": data[i]})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data, label="CPU Power")
    ax.set_ylabel('watts')

    ax2 = ax.twinx()

    # ax2.plot(avg_metric(df, "CPU_TEMP"), label="CPU TEMP (C)", color="red")
    ax2.plot(avg_metric(df, "CPU_USAGE"), label="CPU USAGE (%)", color="orange")
    ax2.plot(df["USED_MEMORY"] * 100 / df["TOTAL_MEMORY"], label="Used Memory (%)", color="green")
    ax2.set_ylim([0, 100])

    ax.set(xlabel=None)
    fig.legend(loc='upper right')
    fig.tight_layout()
    plt.show()


generate_violin_charts("../../../../ENERGY_DATA/sequence_classification")
#generate_metric_charts_csv("../../../../ENERGY_DATA/anakin/quant_energy_data/anakin87-electra-italian-xxl-cased-squad-it_quant_exp00.csv", )
