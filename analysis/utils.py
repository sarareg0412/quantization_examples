from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns;
from optimum.intel import (
    INCModelForSequenceClassification,
    INCModelForQuestionAnswering,
    INCModelForTokenClassification,
    INCModelForMultipleChoice,
    INCModelForMaskedLM,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, SquadExample
)

sns.set()
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

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
INC_tasks = ["INCModelForSequenceClassification", "INCModelForQuestionAnswering", "INCModelForTokenClassification",
             "INCModelForMultipleChoice", "INCModelForMaskedLM", "INCModelForCausalLM", "INCModelForSeq2SeqLM"]
INC_dict = {"INCModelForSequenceClassification": ["text-classification"],
            "INCModelForQuestionAnswering": ["question-answering"],
            "INCModelForTokenClassification": ["token-classification"],
            "INCModelForMultipleChoice": ["multiple-choice"],
            "INCModelForMaskedLM": ["fill-mask"],
            "INCModelForCausalLM": ["text-generation"],
            "INCModelForSeq2SeqLM": ["translation", "conversational", "image-to-text", "summarization"]}
csv_name = "model_data.csv"
csv_header = ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
N_MODELS = 50
SIMPLE_FILTER = False
N_EXPERIMENTS = 0
SEED = 42
QUANT_SPLIT_PERCENT = 0.2  # Quantization split percentage


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


def get_model_from_category(task, model_location):
    model = None
    match task:
        case "INCModelForSequenceClassification":
            model = AutoModelForSequenceClassification.from_pretrained(model_location)
        case "INCModelForQuestionAnswering":
            model = AutoModelForQuestionAnswering.from_pretrained(model_location)
        case "INCModelForTokenClassification":
            model = AutoModelForTokenClassification.from_pretrained(model_location)
        case "INCModelForMultipleChoice":
            model = AutoModelForMultipleChoice.from_pretrained(model_location)
        case "INCModelForMaskedLM":
            model = AutoModelForMaskedLM.from_pretrained(model_location)
        case "INCModelForCausalLM":
            model = AutoModelForCausalLM.from_pretrained(model_location)
        case "INCModelForSeq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_location)

    return model


def get_quantized_model_from_category(task, model_location):
    model = None
    match task:
        case "INCModelForSequenceClassification":
            model = INCModelForSequenceClassification.from_pretrained(model_location)
        case "INCModelForQuestionAnswering":
            model = INCModelForQuestionAnswering.from_pretrained(model_location)
        case "INCModelForTokenClassification":
            model = INCModelForTokenClassification.from_pretrained(model_location)
        case "INCModelForMultipleChoice":
            model = INCModelForMultipleChoice.from_pretrained(model_location)
        case "INCModelForMaskedLM":
            model = INCModelForMaskedLM.from_pretrained(model_location)
        case "INCModelForCausalLM":
            model = INCModelForCausalLM.from_pretrained(model_location)
        case "INCModelForSeq2SeqLM":
            model = INCModelForSeq2SeqLM.from_pretrained(model_location)

    return model


def get_model_from_library(library, category, model_path, quantized=False):
    model = None
    # Switch to retrieve the model class from the different kinds of libraries
    match library:
        case "transformers":
            if quantized:
                model = get_quantized_model_from_category(category, model_path)
            else:
                model = get_model_from_category(category, model_path)
    return model


def get_quantized_model_path(category, model_name):
    # The quantized model is located in the directory like: ./category/model_name_formatted/config
    return os.path.join(os.getcwd(), category, format_name(model_name), "config")


def get_processor_from_category(category, model_name):
    processor = None
    match category:
        case ("INCModelForSequenceClassification" | "INCModelForQuestionAnswering" | "INCModelForTokenClassification" |
              "INCModelForMultipleChoice" | "INCModelForMaskedLM" | "INCModelForCausalLM" | "INCModelForSeq2SeqLM"):
            processor = AutoTokenizer.from_pretrained(model_name)

    return processor


def get_model_data_from_line(line):
    # ['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name']
    model_data_names = csv_header
    model_data_names.append("full_line")
    line = line.split(",")
    model_data = {model_data_names[i]: line[i] for i in range(len(line))}
    return model_data


def get_dataset_from_name(ds_name, ds_config, percent):
    data = (load_dataset(ds_name, ds_config, split="test"))
    data = data.train_test_split(train_size=percent, seed=SEED)["train"]  # Use x% of test dataset
    return data


def create_squad_examples(dataset):
    squad_examples = []

    print("Preprocessing dataset")
    for example in dataset:
        context = example['context']
        question = example['question']
        answer_text = example["answers"]["text"][0]
        start_position = example["answers"]["answer_start"][0]

        squad_example = SquadExample(
            qas_id=example['id'],
            question_text=question,
            context_text=context,
            answer_text=answer_text,
            title=None,
            start_position_character=start_position,
        )

        squad_examples.append(squad_example)

    return squad_examples


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))