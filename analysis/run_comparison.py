import csv
import subprocess

import evaluate
from utils import *

exact_match = evaluate.load("exact_match")


def run_comparison(line):
    model_data = get_model_data_from_line(line)
    if model_data['category'] == 'INCModelForTokenClassification':
        subprocess.run([
            # "../energibridge", "-o", "{}".format(energy_output_file),
            "python", "evaluate_token_classification.py", "{}".format(line)])
    else:
        dataset_file_path = f"INCModelForMaskedLM/{format_name(model_data['model_name'])}/dataset.csv"
        if (os.path.isfile(dataset_file_path)):
            print(f"Reading {dataset_file_path} as dataset")
            data = read_csv(dataset_file_path, ["masked_input", "true_label"], 1)
        else:
            print(f"Loading dataset from hub")
            data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
            data = (data.train_test_split(train_size=TEST_DATA_PERCENT, seed=SEED)["train"])
        # Initialize lists to store references and predictions for accuracy evaluation
        references = get_references(model_data["category"], data, model_data['model_name'])

        print(f"Evaluating Data for model {model_data['model_name']} and "
              f"Dataset {model_data['dataset']}/{model_data['dataset_config_name']}")
        NQ_output = (f"{model_data['category']}/{format_name(model_data['model_name'])}/NQ_output.csv")
        Q_output = (f"{model_data['category']}/{format_name(model_data['model_name'])}/Q_output.csv")

        # Open the CSV file and skip the header row
        with open(NQ_output, 'r') as file:
            csv_reader = csv.reader(file)
            # Read the remaining rows into a list
            nq_predictions = reduce_to_1D_list(list(csv_reader))
        with open(Q_output, 'r') as file:
            csv_reader = csv.reader(file)
            # Read the remaining rows into a list
            q_predictions = reduce_to_1D_list(list(csv_reader))

        if (model_data["category"] == 'INCModelForQuestionAnswering'):
            # Perform exact match score before mapping the nq and q prediction arrays in case the task is question answering
            exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

        nq_predictions = get_predictions(model_data["category"], nq_predictions, references)
        q_predictions = get_predictions(model_data["category"], q_predictions, references)

        if (model_data["category"] != 'INCModelForQuestionAnswering'):
            exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

        # Calculate performance using the loaded metric
        metric = get_metric(model_data["category"])
        NQ_metric = metric.compute(predictions=nq_predictions, references=references)
        Q_metric = metric.compute(predictions=q_predictions, references=references)

        print(f"NQ model metric score is : {NQ_metric}")
        print(f"Q model metric score is : {Q_metric}")
        print(f"Exact match score is : {exact_match_score}")


def get_references(category, data, model_name):
    references = []
    match category:
        case "INCModelForSequenceClassification":
            label_column = "label" if "label" in data.column_names else "labels"
            references = data[label_column]
        case "INCModelForQuestionAnswering":

            references = list(map(lambda example: {"id": example["id"],
                                                   "answers": {"answer_start":example["answers"]["answer_start"],
                                                               "text": [normalize_text(s) for s in example["answers"]["text"]]}},
                                  data))
        case "INCModelForMaskedLM":
            references = data   # The token is already loaded
        case 'INCModelForTokenClassification':
            references = preprocess_tokenized_data(data, model_name)

    return references


def get_predictions(category, prediction, references=None):
    match category:
        case "INCModelForQuestionAnswering":
            prediction = [{"id": references[i]["id"], "prediction_text": normalize_text(example)} for
                          i,example in enumerate(prediction)]
        case 'INCModelForTokenClassification':
            # Turn the string into a real dictionary
            tokens_dict = [ast.literal_eval(pred) for pred in prediction]
            new_pred = [np.zeros((len(ref)), dtype=int) for ref in references]
            for i,pred in enumerate(new_pred):
                for el in tokens_dict[i]:
                    if el['index'] == 13:
                        print('prob')

                    pred[el['index']] = el['ner_tag']
            prediction = new_pred

    return prediction


def get_metric(category):
    metric = None
    match category:
        case "INCModelForSequenceClassification"| "INCModelForMaskedLM":
            metric = evaluate.load("accuracy")
        case "INCModelForQuestionAnswering":
            metric = evaluate.load("squad")
        case "INCModelForTokenClassification":
            metric = evaluate.load("f1")


    return metric
