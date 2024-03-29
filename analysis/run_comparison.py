import csv
import itertools
import subprocess

import evaluate
from utils import *

exact_match = evaluate.load("exact_match")


def run_comparison(line, seed= SEED):
    model_data = get_model_data_from_line(line)
    if model_data['category'] == 'INCModelForToken':
        subprocess.run([
            # "../energibridge", "-o", "{}".format(energy_output_file),
            "python", "evaluate_HF.py", "{}".format(line), 'seqeval'])
    else:
        dataset_file_path = f"INCModelForMaskedLM/{format_name(model_data['model_name'])}/dataset{seed}.csv"
        if model_data['category'] == 'INCModelForMaskedLM' and os.path.isfile(dataset_file_path):
            print(f"Reading {dataset_file_path} as dataset")
            data = read_csv(dataset_file_path, ["masked_input", "true_label"], 1)
        else:
            print(f"Loading dataset from hub")
            data = get_split_dataset(model_data)
            if seed != SEED:
                print(f"DETECTED SEED {seed} FOR COMPARISON")
                data = split_dataset_for_evaluation(data, seed)

        # Initialize lists to store references and predictions for accuracy evaluation
        references = get_references(model_data["category"], data, model_data['model_name'])

        print(f"Evaluating Data for model {model_data['model_name']} and "
              f"Dataset {model_data['dataset']}/{model_data['dataset_config_name']}")
        NQ_output = get_output_file_name(model_data['category'], model_data['model_name'], False)
        Q_output = get_output_file_name(model_data['category'], model_data['model_name'], True)
        print(f"Output files: {NQ_output}; {Q_output}")

        # Open the CSV file and skip the header row
        with open(NQ_output, 'r') as file:
            csv_reader = csv.reader(file)
            # Read the remaining rows into a list
            nq_predictions = reduce_to_1D_list(list(csv_reader))
        with open(Q_output, 'r') as file:
            csv_reader = csv.reader(file)
            # Read the remaining rows into a list
            q_predictions = reduce_to_1D_list(list(csv_reader))

        if model_data["category"] == 'INCModelForQuestionAnswering':
            # Perform exact match score before mapping the nq and q prediction arrays in case the task is question answering
            exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

        nq_predictions = get_predictions(model_data["category"], nq_predictions, references)
        q_predictions = get_predictions(model_data["category"], q_predictions, references)

        if model_data["category"] == 'INCModelForTokenClassification':
            level_lists(q_predictions, nq_predictions)
            nq_predictions = reduce_to_1D_list(nq_predictions)
            q_predictions = reduce_to_1D_list(q_predictions)
            exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)
            print(f"Exact match score is : {exact_match_score}")
            return exact_match_score

        if model_data["category"] != 'INCModelForQuestionAnswering':
            exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

        # Calculate performance using the loaded metric
        metric = get_metric_from_category(model_data["category"])
        NQ_metric = metric.compute(predictions=nq_predictions, references=references)
        Q_metric = metric.compute(predictions=q_predictions, references=references)

        print(f"NQ model metric score is : {NQ_metric}")
        print(f"Q model metric score is : {Q_metric}")
        print(f"Exact match score is : {exact_match_score}")
        return get_final_dict(NQ_metric, Q_metric, exact_match_score)


def get_references(category, data, model_name):
    references = []
    match category:
        case "INCModelForSequenceClassification":
            label_column = "label" if "label" in data.column_names else "labels"
            references = data[label_column]
        case "INCModelForQuestionAnswering":

            references = list(map(lambda example:
                                  {"id": example["id"],
                                   "answers": {"answer_start": example["answers"]["answer_start"],
                                               "text": [normalize_text(s) for s in example["answers"]["text"]]}},
                                  data))
        case "INCModelForMaskedLM":
            references = data[:1001]  # The token is already loaded from the dataset
        case "INCModelForTokenClassification":
            references = preprocess_tokenized_data(data, model_name)
        case "INCModelForMultipleChoice":
            # accuracy needs double quotes
            map_dict = {label: i  for i, label in enumerate(["A", "B", "C", "D"])}
            references = [map_dict[x] for x in data['answer']][:301]

    return references


def get_predictions(category, prediction, references=None):
    match category:
        case "INCModelForQuestionAnswering":
            prediction = [{"id": references[i]["id"], "prediction_text": normalize_text(example)} for
                          i, example in enumerate(prediction)]
        case 'INCModelForTokenClassification':
            # Turn the string into a real dictionary
            tokens_dict = [ast.literal_eval(pred) for pred in prediction]
            prediction = tokens_dict

    return prediction


def get_final_dict(NQ_metric, Q_metric, exact_match):
    results = exact_match
    results = add_new_value('NQ','accuracy', NQ_metric, results)
    results = add_new_value('NQ','f1', NQ_metric, results)
    results = add_new_value('Q','accuracy', Q_metric, results)
    results = add_new_value('Q','f1', Q_metric, results)

    return results


def add_new_value(prefix, metric, old_dict, new_dict):
    for key in old_dict.keys():
        # Check if the key starts with the specified prefix
        if key.startswith(metric):
            new_dict[f'{prefix}_{metric}'] = old_dict[key]
    return new_dict
