import csv

import evaluate
from utils import *

exact_match = evaluate.load("exact_match")


def run_comparison(model_data):
    dataset_file_path = f"INCModelForMaskedLM/{format_name(model_data['model_name'])}/dataset.csv"
    if (os.path.isfile(dataset_file_path)):
        print(f"Reading {dataset_file_path} as dataset")
        data = read_csv(dataset_file_path, ["masked_input", "true_label"], 1)
    else:
        print(f"Loading dataset from hub")
        data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
        data = (data.train_test_split(train_size=TEST_DATA_PERCENT, seed=SEED)["train"])
    # Initialize lists to store references and predictions for accuracy evaluation
    references = get_references(model_data["category"], data)

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

    exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

    nq_predictions = get_predictions(model_data["category"], nq_predictions, references)
    q_predictions = get_predictions(model_data["category"], q_predictions, references)
    # Calculate accuracy using the loaded accuracy metric

    metric = get_metric(model_data["category"])
    NQ_metric = metric.compute(predictions=nq_predictions, references=references)
    Q_metric = metric.compute(predictions=q_predictions, references=references)

    print(f"NQ model metric score is : {NQ_metric}")
    print(f"Q model metric score is : {Q_metric}")
    print(f"Exact match score is : {exact_match_score}")


def get_references(category, data):
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

    return references


def get_predictions(category, prediction, references=None):
    match category:
        case "INCModelForQuestionAnswering":
            prediction = [{"id": references[i]["id"], "prediction_text": normalize_text(example)} for
                          i,example in enumerate(prediction)]

    return prediction


def get_metric(category):
    metric = None
    match category:
        case "INCModelForSequenceClassification"| "INCModelForMaskedLM":
            metric = evaluate.load("accuracy")
        case "INCModelForQuestionAnswering":
            metric = evaluate.load("squad")

    return metric
