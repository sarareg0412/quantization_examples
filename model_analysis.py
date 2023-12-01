import csv
from collections import ChainMap
from distutils.command.config import config
from functools import reduce
from utils import categories, category_dict
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import numpy
from huggingface_hub import HfApi, ModelFilter, DatasetFilter  # api to interact with the hub

N_MODELS = 5
hf_api = HfApi()
SIMPLE_FILTER = False
category = categories[3]

def add_properties(model, task):
    model.task = task
    # model.ratio = model.likes/model.downloads
    return model


# For each task we get the top N_MODELS models with most likes and downloads
def get_models_of_task(task):
    models = list(map(lambda x: add_properties(x, task),
                            hf_api.list_models(
                                filter=ModelFilter(
                                    task=task,
                                ),
                                sort="likes",       # Values are the properties of the huggingface_hub.hf_api.ModelInfo class.
                                direction=-1,       # sort by descending order
                                limit=N_MODELS*N_MODELS,  # The limit on the number of models fetched.
                                cardData=True,      # Whether to grab the metadata for the model as well. Can contain useful information such as
                                                    # carbon emissions, metrics, and datasets trained on.
                            )
                        ))
    # Further filtering because the "task" filter is sometimes not the same as the pipeline_tag, which is what is
    # actually used to filter the models on the website.
    return list(filter(lambda x: x.task == x.pipeline_tag, models))


def check_existing_dataset(model):
    # They said it would work with a list of strings as dataset_name but it doesn't
    # For all datasets used to train the model we retrieve its info.
    for dataset in model.cardData["datasets"]:
        ds = list(hf_api.list_datasets(
            filter=DatasetFilter(
                dataset_name=dataset
            ),
            limit=1,
            full=True
        ))

        if SIMPLE_FILTER:
            # We return the first dataset that matches the given name without further investigations on it
            if len(ds) == 1 and ds[0].id == dataset:
                model.dataset = dataset
                return True
        else:
            # If we retrieved one dataset and its name matches the one on the model's metadata we return True:
            # that means we found the exact dataset used to train the model
            if len(ds) == 1 and ds[0].id == dataset:
                datasetInfo = ds[0]
                # Check whether the cardData and 'dataset_info' attributes exist
                if (getattr(datasetInfo, "cardData", None) is not None) and ('dataset_info' in datasetInfo.cardData):
                    # We check whether the dataset has multiple configurations. If so, we have to loop through them
                    if isinstance(datasetInfo.cardData['dataset_info'], list):
                        for config in datasetInfo.cardData['dataset_info']:
                            # We look for the "test" split of the dataset configuration and if there is one we return
                            # True because we only need one dataset with the test split.
                            for split in config['splits']:
                                if split['name'] == 'test':
                                    model.dataset = dataset
                                    model.ds_config_name = config['config_name']
                                    return True
                    else:
                        # The dataset doesn't have multiple configurations, we look for the "test" split.
                        for split in datasetInfo.cardData['dataset_info']['splits']:
                            if split['name'] == 'test':
                                model.dataset = dataset
                                return True

    return False


def get_top_models_of_category(list_tasks, n):
    # Retrieving the list of models for all tasks in the category
    models = [get_models_of_task(task) for task in list_tasks]
    # We reduce the list of models of different tasks per category, in a single 1D list
    models = reduce(lambda x, y: x + y, models, [])
    # Take out duplicate models since the same model could belong to different tasks of the same category
    seen = set()
    models = [mod for mod in models if mod.id not in seen and not seen.add(mod.id)]

    # Select only models that have datasets info in their cardData metadata
    models = list(filter(lambda x: ("datasets" in x.cardData),
                         filter(lambda x: (getattr(x, "cardData", None) is not None), models)))
    # Filter out models that don't have datasets on huggingface
    models = list(filter(lambda model: check_existing_dataset(model), models))
    # Sort based on our own metric
    models.sort(key=lambda x: x.likes, reverse=True)
    top_n_models = list(map(lambda model: [model.modelId, model.likes, model.downloads
                                        , model.task, getattr(model, "library_name", "")
                                        , model.dataset,  getattr(model, "ds_config_name", "")], models[:n]))
    return top_n_models

def create_csv(models):
    csv_file = 'model_data.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Model name', 'Likes', 'Downloads', 'Task', 'Library', 'Dataset', 'Dataset Config name'])
        # Write the remaining rows
        writer.writerows(models)

        #for category in models:
        #    writer.writerows(category)


# We get the list of top models for a list of tasks of the same category.
models = get_top_models_of_category(list_tasks=category_dict[category], n=N_MODELS)
# We filter out models with 0 likes or downloads and turn everything into a list. We add the
# "ratio" parameter to the model, representing the ratio between #likes/#downloads
create_csv(models)
print("Top {} models for category {}.".format(N_MODELS, category))
print("{:<50} {:<10} {:<15} {:<30} {:<25} {:<15} {:<15}".format('Model name','Likes','Downloads','Task', 'Library', 'Dataset', "Config"))
for model in models:
    print(model)
