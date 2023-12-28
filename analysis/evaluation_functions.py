import torch
from transformers import pipeline
from utils import  get_processor_from_category

def eval_func(model, dataset, metric):
    # Assume the object column name is the first one and the label column name is the last one
    obj_name = dataset.column_names[0]
    label_name = dataset.column_names[-1]
    trainset = dataset.select_columns([obj_name, label_name])
    trainset = trainset.map(lambda e: {'obj': e[obj_name], 'labels': e[label_name]})
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=4)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    pipe = pipeline( model=model, image_processor=processor)
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