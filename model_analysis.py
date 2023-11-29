from functools import reduce
from utils import categories, category_dict
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import numpy
from huggingface_hub import HfApi, ModelFilter, DatasetFilter  # api to interact with the hub

N_MODELS = 5
hf_api = HfApi()

category = categories[1]


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
                                sort="likes",  # Values are the properties of the huggingface_hub.hf_api.ModelInfo class.
                                direction=-1,  # sort by descending order
                                limit=N_MODELS*N_MODELS,  # The limit on the number of models fetched.
                                cardData=True  # Whether to grab the metadata for the model as well. Can contain useful information such as
                                # carbon emissions, metrics, and datasets trained on.
                            )
                        ))
    # Further filtering because the "task" filter is sometimes not the same as the pipeline_tag, which is what is
    # actually used to filter the models on the website.
    return list(filter(lambda x: x.task == x.pipeline_tag , models))


def check_existing_dataset(model):
    # They said it would work with a list of strings as dataset_name but it doesn't
    model_datasets = []
    for dataset in model.cardData["datasets"]:
        ds = list(hf_api.list_datasets(
            filter=DatasetFilter(
                dataset_name=dataset
            ),
            limit=1
        ))
        # If we retrieved one dataset and its name matches the one on the model's metadata we return True:
        # that means we found the exact dataset used to train the model
        if len(ds) == 1 and ds[0].id == dataset:
            model.dataset = dataset
            return True

    return False

def normalize_models(models):
    data = list(map(lambda x: x.downloads,models))
    scaler = MinMaxScaler()
    scaler.fit(data)
    res = scaler.transform(data)
    print(res)
    #models_scaled = minmax_scale(models[['likes','downloads']], feature_range=(0,1))
    return 1

def get_top_models_of_category(list_tasks, n):
    # Retrieving the list of models for all tasks in the category
    models = [get_models_of_task(task) for task in list_tasks]
    # We reduce the list of models of different tasks per category, in a single 1D list
    models = reduce(lambda x, y: x + y, models, [])
    # Take out duplicate models since the same model could belong to different tasks of the same category
    seen = set()
    models = [mod for mod in models if mod.id not in seen and not seen.add(mod.id)]
    # TODO Normalize phase
    #normalize_models(models)

    # Select only models that have datasets info in their cardData metadata
    models = list(filter(lambda x: ("datasets" in x.cardData),
                         filter(lambda x: (getattr(x, "cardData", None) is not None), models)))
    # Filter out models that don't have datasets on huggingface
    models = list(filter(lambda model: check_existing_dataset(model), models))
    # TODO Sort based on our own metric
    models.sort(key=lambda x: x.likes, reverse=True)
    return models[:n]


# We get the list of top models for a list of tasks of the same category.
models = get_top_models_of_category(list_tasks=category_dict[category], n=N_MODELS)
# We filter out models with 0 likes or downloads and turn everything into a list. We add the
# "ratio" parameter to the model, representing the ratio between #likes/#downloads

print("Top {} models for category {}.".format(N_MODELS, category))
print("{:<50} {:<10} {:<15} {:<30} {:<20} {:<10}".format('Model name','Likes','Downloads','Task', 'Library', 'Dataset'))
for model in models:
    print("{:<50} {:<10} {:<15} {:<30} {:<20} {:<10}".format(model.modelId, model.likes, model.downloads
                                          , model.task, getattr(model, "library_name", "None")
                                                             , model.dataset ))
