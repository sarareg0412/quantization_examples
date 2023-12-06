import csv
from utils import *
from huggingface_hub import HfApi, ModelFilter, DatasetFilter  # api to interact with the hub

hf_api = HfApi()


def add_properties(model, task):
    model.task = task
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


def model_has_test_datasets(model):
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
            # If we retrieved one dataset and its name matches the one on the model's metadata that means we found
            # the exact dataset used to train the model
            if len(ds) == 1 and ds[0].id == dataset:
                datasetInfo = ds[0]
                # Check whether the "cardData", 'dataset_info' attributes of the dataset exist
                if (getattr(datasetInfo, "cardData", None) is not None) and ('dataset_info' in datasetInfo.cardData):
                    # We check whether the dataset has multiple configurations. If so, we have to loop through them
                    if isinstance(datasetInfo.cardData['dataset_info'], list):
                        for config in datasetInfo.cardData['dataset_info']:
                            # We look for the "test" split of the dataset configuration and if there is one we return
                            # True because we only need one dataset with the test split.
                            try:
                                for split in config['splits']:
                                    if split['name'] == 'test':
                                        model.dataset = dataset
                                        model.ds_config_name = config['config_name']
                                        return True
                            except Exception as e:
                                print(e)

                    else:
                        try:
                            # The dataset doesn't have multiple configurations, we look for the "test" split.
                            for split in datasetInfo.cardData['dataset_info']['splits']:
                                if split['name'] == 'test':
                                    model.dataset = dataset
                                    return True
                        except Exception as e:
                            print(e)

    return False


def check_existing_dataset(models):
    # They said it would work with a list of strings as dataset_name but it doesn't
    best_models = []
    for model in models:
        if model_has_test_datasets(model):
            best_models.append(model)
        # if we already got N_MODELS, we return the list
        if len(best_models) >= N_MODELS:
            return best_models

    return best_models


def get_top_models_of_category(category, n):
    # Retrieving the list of models for all tasks in the category
    models = [get_models_of_task(task) for task in category_dict[category]]
    # We reduce the list of models of different tasks per category, in a single 1D list
    models = reduce_to_1D_list(models)
    # Take out duplicate models since the same model could belong to different tasks of the same category
    seen = set()
    models = [mod for mod in models if mod.id not in seen and not seen.add(mod.id)]

    # Select only models that have datasets info in their cardData metadata
    models = list(filter(lambda x: ("datasets" in x.cardData),
                         filter(lambda x: (getattr(x, "cardData", None) is not None), models)))
    # Sort based on our own metric
    models.sort(key=lambda x: x.likes, reverse=True)
    # Get only top N models that have a test datasets on huggingface, most intensive task
    category_top_n_models = check_existing_dataset(models)
    # Map models to the correct format for the CSV
    category_top_n_models = list(map(lambda model: [model.modelId, model.likes, model.downloads
                                            , category, model.task, getattr(model, "library_name", "")
                                            , model.dataset,  getattr(model, "ds_config_name", "")], category_top_n_models))
    return category_top_n_models


def create_csv(models_list, category):
    csv_file = ""
    if category is not None:
        csv_file = "{}_".format(category)
    csv_file += csv_name

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['model_name', 'likes', 'downloads', 'category', 'task', 'library', 'dataset', 'dataset_config_name'])
        # Write the remaining rows
        writer.writerows(models_list)


def create_categories_csv():
    for i in range(0,N_CATEGORIES):
        # We get the list of top models for a list of tasks of the same category.
        category = categories[i]
        print("Get top {} models of category {}".format(N_MODELS, category))
        cat_models = get_top_models_of_category(category=category, n=N_MODELS)
        print("Start CSV creation for category {}".format(category))
        create_csv(cat_models, category)
        print("Done.")


def create_full_csv():
    models = []
    for i in range(0,N_CATEGORIES):
        # We get the list of top models for a list of tasks of the same category.
        category = categories[i]
        print("Get top {} models of category {}".format(N_MODELS, category))
        models.extend(get_top_models_of_category(category=category, n=N_MODELS))

    print("Start CSV creation")
    create_csv(models, None)
    print("Done.")


create_categories_csv()