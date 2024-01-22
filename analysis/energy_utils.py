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
        files = os.listdir(path)
        sorted_files = sorted(files, key=lambda x: x.split('_')[2])
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


# generate_metric_charts("../../../../ENERGY_DATA/anakin/quant_energy_data", "anakin87_electra-italian-xxl-cased")
# generate_metric_charts_csv("../../../../ENERGY_DATA/anakin/quant_energy_data/anakin87-electra-italian-xxl-cased-squad-it_quant_exp00.csv", )
