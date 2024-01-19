import ast
import csv
from functools import reduce
import random

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from optimum.intel import (
    INCModelForSequenceClassification,
    INCModelForQuestionAnswering,
    INCModelForTokenClassification,
    INCModelForMultipleChoice,
    INCModelForMaskedLM,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM
)
from torch.utils.data import Dataset
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
TEST_DATA_PERCENT = 0.05


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def reduce_to_1D_list(list):
    return reduce(lambda x, y: x + y, list, [])


def format_name(name):
    return name.replace("/", "-")


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

    for example in dataset:
        context = example['context']
        question = example['question']
        answer_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ''
        start_position = example["answers"]["answer_start"][0] if len(example["answers"]["answer_start"]) > 0 else None

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


# generate_metric_charts("../../../../ENERGY_DATA/cardiff-latest/quant_energy_data")
# Masked LM
def create_maskedlm_examples2(data, model_name):
    dataset_file_path = f"INCModelForMaskedLM/{format_name(model_name)}/dataset.csv"
    tokenizer = get_processor_from_category("INCModelForMaskedLM", model_name)
    if (os.path.isfile(dataset_file_path)):
        print(f"Reading {dataset_file_path} as dataset")
        dataset = read_csv(dataset_file_path, ["masked_input", "true_label"])
        # Convert the string of tokens into an array
        dataset = [ast.literal_eval(line) for line in dataset]
        # Decode the list of tokens as a dataset
        dataset = [tokenizer.decode(line) for line in dataset]
        return dataset
    else:
        random.seed(10)
        mask_token = tokenizer.mask_token_id

        def tokenize_function(examples):
            return tokenizer(examples["whole_func_string"])

        data_tokenized = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

        # TODO set block_size = tokenizer.model_max_length
        block_size = 128

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        data_tokenized = data_tokenized.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

        labels = []
        def mask_random_token(example):
            tokens = example["input_ids"]
            random_index = np.random.randint(0, len(tokens))
            # Add unmasked lable
            tokens[random_index] = mask_token
            # Adding the correct decoded label of the masked token
            labels.append(example["labels"][random_index])
            example["input_ids"] = tokens
            return example

        # Then mask one of the input data tokens
        data_tokenized = [mask_random_token(e) for e in data_tokenized]

        dataset = [[data_tokenized[i]["input_ids"], labels[i]] for i in range(len(data_tokenized))]
        write_csv(dataset_file_path,dataset, ["masked_input", "true_label"])

        data_tokenized = [tokenizer.decode(x["input_ids"]) for x in data_tokenized]
        return data_tokenized


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def write_csv(output_file_name, content, header=None):
    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(content)


def read_csv(file_name, header=None, column_index=0):
    with open(file_name) as file:
        csvReader = csv.reader(file)
        if header is not None:
            csvReader.__next__()  # Skip header
        # Read content
        content = [line[column_index] for line in csvReader]

    return content
