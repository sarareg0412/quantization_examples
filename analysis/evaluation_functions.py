import evaluate
from transformers import pipeline
from utils import *


def eval_func_with_metric(model, dataset, metric):
    # Assume the object column name is the first one and the label column name is the last one
    obj_name = dataset.column_names[0]
    label_name = dataset.column_names[-1]
    trainset = dataset.select_columns([obj_name, label_name])
    trainset = trainset.map(lambda e: {'obj': e[obj_name], 'labels': e[label_name]})
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=4)
    # processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    # pipe = pipeline( model=model, image_processor=processor)
    for batch in dataloader:
        obj_tensor = batch[obj_name]
        label_tensor = batch[label_name]
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        output = model(obj_tensor)
        predicted_label = output[0]['label'] if isinstance(output, list) \
            else output['label']
        metric.update(predicted_label, label_tensor)

    accuracy = metric.result()
    return accuracy


def eval_func(model, dataset):
    pipe = pipeline(model=model)
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    predictions = []
    # Iterate through the test split
    for object in dataset:
        # Load object and label truth label from the dataset
        object = object[dataset.column_names[0]]  # Assume the object column name is the first one
        label = object[dataset.column_names[-1]]  # Assume the label column name is the last one

        # Infer the object label using the model
        prediction = pipe(object)
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        prediction = prediction[0]['label'] if isinstance(prediction, list) else prediction['label']

        # Append ground truth label and predicted label for "accuracy" evaluation
        references.append(label)
        predictions.append(model.config.label2id[prediction])  # Map the predicted label using the model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    exact_match = evaluate.load("exact_match")
    exact_match_score = exact_match.compute(predictions=predictions, references=references)
    return exact_match_score
