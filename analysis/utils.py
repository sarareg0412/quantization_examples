import ast
import csv
from functools import reduce
import random

import os
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

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import init_empty_weights

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
TEST_DATA_PERCENT = 0.5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def reduce_to_1D_list(list):
    return reduce(lambda x, y: x + y, list, [])


def format_name(name):
    return name.replace("/", "-")


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


# Question answering
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
def create_maskedlm_examples(data, model_name):
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
        write_csv(dataset_file_path, dataset, ["masked_input", "true_label"])

        data_tokenized = [tokenizer.decode(x["input_ids"]) for x in data_tokenized]
        return data_tokenized


# Token Classification
def create_tokenclass_examples(data, model_name):
    tokenizer = get_processor_from_category("INCModelForTokenClassification", model_name)

    def decode_data(example):
        return tokenizer.convert_tokens_to_string(example['tokens'])

    data_tokenized = [decode_data(ex) for ex in data]
    return data_tokenized


def preprocess_tokenized_data(data, model_name):
    tokenizer = get_processor_from_category("INCModelForTokenClassification", model_name)

    def decode_tokens(example):
        example['word_ids'] = tokenizer(example["tokens"], is_split_into_words=True).word_ids()
        return example

    # data = data.map(decode_tokens)

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    data = data.map(tokenize_and_align_labels)

    new_labels = []

    return new_labels


# Multiple Choice
def get_multichoice_output(data, model, model_name):
    tokenizer = get_processor_from_category('INCModelForMultipleChoice', model_name)
    labels = torch.tensor(0).unsqueeze(0)
    MAX_TOKEN_LENGTH = tokenizer.model_max_length-4
    questionTokens = tokenizer.encode(data['question'], truncation=True, max_length=MAX_TOKEN_LENGTH)[1:-1]
    contextTokens = tokenizer.encode(data['article'], truncation=True, max_length=MAX_TOKEN_LENGTH)[1:-1]
    answers = data['options']
    logits_sum = torch.zeros(1, len(answers))

    maxAns = 0
    for x in answers:
        tokenLen = len(tokenizer.encode(x, truncation=True, max_length=MAX_TOKEN_LENGTH)) - 2
        if tokenLen > maxAns:
            maxAns = tokenLen
    remainingTokens = len(contextTokens)
    while remainingTokens > 0:
        tokensToRemove = 0
        totalMaxTokens = len(questionTokens) + len(contextTokens) + maxAns + 3
        if totalMaxTokens > MAX_TOKEN_LENGTH:
            tokensToRemove = totalMaxTokens - MAX_TOKEN_LENGTH
            choppedContext = contextTokens[:-tokensToRemove]
        else:
            choppedContext = contextTokens
        remainingTokens = remainingTokens - len(choppedContext)
        contextTokens = contextTokens[-tokensToRemove:]
        ANS = [[tokenizer.decode(choppedContext) + tokenizer.decode(questionTokens), candidate] for candidate in
               answers]
        inputs = tokenizer(ANS, return_tensors="pt", padding='max_length')
        with torch.no_grad():
            outputs = (model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)).__dict__
        logits_sum += outputs['logits']

    # Return the number associated with the class
    return logits_sum.argmax().item()


def level_lists(l1, l2):
    for i in range(len(l1)):
        difference = len(l1[i]) - len(l2[i])
        if difference < 0:
            l1[i].extend(['0'] * abs(difference))
        elif difference > 0:
            l2[i].extend(['0'] * difference)


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
        if header is not None:
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


def convert_to_nums(path):
    content = read_csv(path)
    map_dict = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
    content_mapped = [[map_dict[x]] for x in content]
    new_filename = os.path.splitext(path)[0] + '_copy.csv'
    write_csv(new_filename,content_mapped)
