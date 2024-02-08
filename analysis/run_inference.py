import csv
import sys

from tqdm import tqdm as t
from transformers.pipelines.pt_utils import KeyDataset

from utils import *
from transformers import pipeline
from datasets import load_dataset


def run_inference_from_line(is_quantized, line, train_size=TEST_DATA_PERCENT, seed=SEED):
    model_data = get_model_data_from_line(line)
    data = get_split_dataset(model_data, train_size=train_size, seed=seed)

    # map the dataset based on the category
    data = map_data(data, model_data)

    quantized = True if (is_quantized == "True") else False
    model_path = get_quantized_model_path(model_data["category"], model_data["model_name"])
    # If we want to evaluate the NOT quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model_path = model_data["model_name"]

    model = get_model_from_library(model_data["library"], model_data["category"], model_path, quantized=quantized)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    output_file_name = (f"{model_data['category']}/{format_name(model_data['model_name'])}/"
                        f"{'' if quantized else 'N'}Q_output.csv")

    print(f"PERFORMING INFERENCE on {'QUANTIZED ' if quantized else ' '}{model_data['model_name']} and "
          f"{model_data['dataset']}/{model_data['dataset_config_name']}")

    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        if model_data["category"] == 'INCModelForMultipleChoice':
            for i in t(range(len(data))):
                out = get_multichoice_output(data[i], model, model_data['model_name'])
                # Since there might be multiple labels with multiple scores associated, we get the first one.
                prediction = get_prediction(out, model_data["category"])
                writer.writerow(prediction)
        else:
            if model_data["category"] == 'INCModelForTokenClassification':
                pipe = pipeline(model_data["task"], model=model, tokenizer=processor, grouped_entities=True)
            else:
                pipe = pipeline(model_data["task"], model=model, tokenizer=processor)
            for out in t(pipe(data), total=len(data)):
                # Since there might be multiple labels with multiple scores associated, we get the first one.
                prediction = get_prediction(out, model_data["category"], model.config.label2id)
                writer.writerow(prediction)
        print("Done")


def map_data(data, model_data):
    category = model_data["category"]
    print("Preprocessing dataset...")
    match category:
        case "INCModelForSequenceClassification":
            data = KeyDataset(data, "text")
        case "INCModelForQuestionAnswering":
            data = ListDataset(create_squad_examples(data))
        case "INCModelForMaskedLM":
            data = ListDataset(create_maskedlm_examples(data, model_data["model_name"]))
        case "INCModelForTokenClassification":
            data = ListDataset(create_tokenclass_examples(data, model_data['model_name']))
        case "INCModelForMultipleChoice":
            data = ListDataset(data)

    print("Done.")
    return data


def get_prediction(out, category, convert_fn=None):
    res = []
    match category:
        case "INCModelForSequenceClassification":
            result = out[0]['label'] if isinstance(out['label'], list) else out['label']
            res.append(convert_fn[result])
        case "INCModelForQuestionAnswering":
            result = out[0]["answer"] if isinstance(out, list) else out["answer"]
            res.append(result)
        case "INCModelForMaskedLM":
            res.append(out[0]["token"])
        case "INCModelForTokenClassification":
            res.append([res['entity_group'] for res in out])
        case "INCModelForMultipleChoice":
            # No need to convert the output in a label (0:A, 1:B ecc) because HF accuracy gives
            # error when letters instead of numbers are given
            res.append(out)
    return res


if __name__ == "__main__":
    run_inference_from_line(sys.argv[1], sys.argv[2])
