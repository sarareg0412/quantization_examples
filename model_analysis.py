from functools import reduce


from sklearn.preprocessing import minmax_scale, MinMaxScaler
import numpy
from huggingface_hub import HfApi, ModelFilter, DatasetFilter  # api to interact with the hub

N_MODELS = 5
hf_api = HfApi()
categories = ["multi-modal", "computer-vision", "natural-language-processing", "audio", "tabular",
              "reinforcement-learning"]
category = categories[1]
# The dictionary has categories as keys and the list of tasks associated with them as values.
category_dict = {"multi-modal": ["feature-extraction", "text-to-image", "image-to-text", "text-to-video",
                                 "visual-question-answering",
                                 "document-question-answering", "graph-machine-learning"],
                 "computer-vision": ["depth-estimation", "image-classification", "object-detection",
                                     "image-segmentation",
                                     "image-to-image", "unconditional-image-generation", "video-classification",
                                     "zero-shot-image-classification"],
                 "natural-language-processing": ["text-classification", "token-classification",
                                                 "table-question-answering",
                                                 "question-answering", "zero-shot-classification", "translation",
                                                 "summarization",
                                                 "conversational", "text-generation", "text2text-generation",
                                                 "fill-mask", "sentence-similarity"],
                 "audio": ["text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio",
                           "audio-classification",
                           "voice-activity-detection"],
                 "tabular": ["tabular-classification", "tabular-regression"],
                 "reinforcement-learning": ["reinforcement-learning", "robotics"]
                 }


def add_properties(model, task):
    model.task = task
    # model.ratio = model.likes/model.downloads
    return model


# For each task we get the top N_MODELS models with most likes and downloads
def get_models_of_task(task):
    return list(map(lambda x: add_properties(x, task), hf_api.list_models(
        filter=ModelFilter(
            task=task,
        ),
        sort="likes" and "downloads",  # Values are the properties of the huggingface_hub.hf_api.ModelInfo class.
        direction=-1,  # sort by descending order
        limit=N_MODELS*N_MODELS,  # The limit on the number of models fetched.
        cardData=True  # Whether to grab the metadata for the model as well. Can contain useful information such as
        # carbon emissions, metrics, and datasets trained on.
    )))


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
        # Add the dataset only if its name matches the one on the model's metadata
        if len(ds) == 1 and ds[0].id == dataset:
            model_datasets.append(ds)

    return len(model_datasets) > 0

def normalize_models(models):
    data = list(map(lambda x: x.downloads,models))
    scaler = MinMaxScaler()
    scaler.fit(data)
    res = scaler.transform(data)
    print(res)
    #models_scaled = minmax_scale(models[['likes','downloads']], feature_range=(0,1))
    return 1

def get_models_of_category(list_tasks):
    # We reduce the list of models of different tasks per category, in a single 1D list
    models = reduce(lambda x, y: x + y, [get_models_of_task(task) for task in list_tasks], [])
    # Take out duplicate models since the same model could belong to different tasks of the same category
    seen = set()
    models = [mod for mod in models if mod.id not in seen and not seen.add(mod.id)]
    # TODO Normalize phase
    normalize_models(models)
    # TODO Sort based on our own metric
    models.sort(key=lambda x: x.likes, reverse=True)

    # Select only models that have datasets info in their cardData metadata
    models = list(filter(lambda x: ("datasets" in x.cardData),
                         filter(lambda x: (getattr(x, "cardData", None) is not None), models)))
    # Filter out models that don't have datasets on huggingface
    models = list(filter(lambda model: check_existing_dataset(model), models))
    return models


# We get the list of top models for a list of tasks of the same category.
models = get_models_of_category(list_tasks=category_dict[category])
# We filter out models with 0 likes or downloads and turn everything into a list. We add the
# "ratio" parameter to the model, representing the ratio between #likes/#downloads

print("Top {} models for category {}.".format(N_MODELS, category))
print("Model name\t\t\t\t\tlikes\tdownloads\tlibrary\ttask\tdatasets")
for model in models[:N_MODELS]:
    print("{}\t{}\t{}\t{}\t{}\t{}".format(model.modelId, model.likes, model.downloads,
                                          getattr(model, "library_name", "None")
                                          , model.task, model.cardData["datasets"], ))
