from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns;

sns.set()
import numpy as np
import torch

from transformers import AutoModelForImageClassification

N_CATEGORIES = 6
# The dictionary has categories as keys and the list of tasks associated with them as values.
category_dict = {"multi-modal": ["feature-extraction", "text-to-image", "image-to-text", "text-to-video",
                                 "visual-question-answering",
                                 "document-question-answering", "graph-machine-learning"],
                 "computer-vision": ["depth-estimation", "image-classification", "object-detection",
                                     "image-segmentation",
                                     "image-to-image", "unconditional-image-generation", "video-classification",
                                     "zero-shot-image-classification"],
                 "natural-language-processing": ["text-classification", "token-classification",
                                                 "table-question-answering",
                                                 "question-answering", "zero-shot-classification", "translation",
                                                 "summarization",
                                                 "conversational", "text-generation", "text2text-generation",
                                                 "fill-mask", "sentence-similarity"],
                 "audio": ["text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio",
                           "audio-classification",
                           "voice-activity-detection"],
                 "tabular": ["tabular-classification", "tabular-regression"],
                 "reinforcement-learning": ["reinforcement-learning", "robotics"]
                 }
categories = ["multi-modal", "computer-vision", "natural-language-processing", "audio", "tabular",
              "reinforcement-learning"]
csv_name = "model_data.csv"
csv_header = ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
N_MODELS = 5
SIMPLE_FILTER = False
N_EXPERIMENTS = 5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def reduce_to_1D_list(list):
    return reduce(lambda x, y: x + y, list, [])


def format_name(name):
    return name.replace("/", "-")


def generate_metric_charts(path: str, model_name, workloads: [str]):
    fig, ax = plt.subplots(figsize=[5, 3])
    ax.set_ylim(0, 60)
    ax.set_xlim(0, 100)

    for workload in workloads:
        all_data = []
        if not os.path.isdir(path):
            continue
        for csv_file in os.listdir(path):
            if not csv_file.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(path, csv_file))
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
            # data = data[1:-1] # take out first experiment
            for i in range(0, len(data)):
                all_data.append({"Time": i, "CPU_POWER (Watts)": data[i]})

        plot = sns.lineplot(data=pd.DataFrame(all_data), x="Time", y="CPU_POWER (Watts)", estimator=np.median,
                            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)), ax=ax, legend=True,
                            label=workload.title())
        plot.set(xlabel=None, ylabel=None)
        title = f"{model_name}_energy_data_plot.pdf"
        plot.set_title(title)

    plot.get_figure().savefig(os.path.join(f"{path}/{title}"))
    # plt.show()


def get_model_from_task(task, model_location):
    model = None
    match task:
        case "image-classification":
            model = AutoModelForImageClassification.from_pretrained(model_location)

    return model
