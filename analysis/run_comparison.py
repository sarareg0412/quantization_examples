import csv

import evaluate
from utils import *

exact_match = evaluate.load("exact_match")


def run_comparison(model_data):
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = (data.train_test_split(train_size=TEST_DATA_PERCENT, seed=SEED)["train"])
    # .select(range(500))) # Use 50% of test dataset to run comparison
    # Get processor (image processor, tokenizer etc.)
    # processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    # No need to retrieve the non quantized model as we only need its name to retrieve it from the hub
    # Retrieve quantized model by its configuration.
    # nq_model = get_model_from_library(model_data["library"], model_data["category"], model_data["model_name"])
    # q_model = get_model_from_library(model_data["library"], model_data["category"],
    #                                 get_quantized_model_path(model_data["category"], model_data["model_name"]),
    #                                 quantized=True
    #                                 )
    # Setup non quantized and quantized model pipeline for inference
    # nq_pipe = pipeline(model_data["task"], model=nq_model, tokenizer=processor)
    # q_pipe = pipeline(model_data["task"], model=q_model, tokenizer=processor)
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
            references = list(map(lambda example: {"id": example["id"], "answers": example["answers"]},
                                  data))

    return references


def get_predictions(category, prediction, references=None):
    match category:
        case "INCModelForQuestionAnswering":
            prediction = [{"id": references[i]["id"], "prediction_text": example} for i,example in enumerate(prediction)]

    return prediction


def get_metric(category):
    metric = None
    match category:
        case "INCModelForSequenceClassification":
            metric = evaluate.load("accuracy")
        case "INCModelForQuestionAnswering":
            metric = evaluate.load("squad")

    return metric
