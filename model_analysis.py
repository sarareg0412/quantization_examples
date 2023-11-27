from functools import reduce

import numpy
from huggingface_hub import HfApi, ModelFilter  # api to interact with the hub

N_MODELS = 5
hf_api = HfApi()

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
    # model.ratio = model.likes/model.downloads
    model.task = task
    return model


# For each task we get the top N_MODELS models with most likes and downloads
def get_models_of_task(task):
    return list(map(lambda x: add_properties(x, task), hf_api.list_models(
        filter=ModelFilter(
            task=task,
        ),
        sort="likes" and "downloads",  # Values are the properties of the huggingface_hub.hf_api.ModelInfo class.
        direction=-1,    # sort by descending order
        limit=N_MODELS,  # The limit on the number of models fetched.
        cardData=True    # Whether to grab the metadata for the model as well. Can contain useful information such as
                         # carbon emissions, metrics, and datasets trained on.
    )))


def get_models_of_category(list_tasks):
    # We reduce the list of models of different categories in a single 1D list
    models = reduce(lambda x, y: x + y, [get_models_of_task(task) for task in list_tasks], [])
    #Select only models that have datasets in their metadata
    models = list(filter(lambda x: (getattr(x, "cardData", None) is not None), models))
    #TODO
    models = list(filter(lambda x: (getattr(x.cardData, "dataset", None) is not None), models))
    # Return all unique models since the same models could belong to different tasks
    seen = set()
    models = [mod for mod in models if mod.id not in seen and not seen.add(mod.id)]

# We get the list of top models for a list of tasks of the same category.
models = get_models_of_category(list_tasks=category_dict["computer-vision"])
# We filter out models with 0 likes or downloads and turn everything into a list. We add the
# "ratio" parameter to the model, representing the ratio between #likes/#downloads
# Sorting phase based on likes,downloads or ratio. TBD with our own metric
models.sort(key=lambda x: x.likes, reverse=True)
print("Top {} models. Total number of unique models: {}".format(N_MODELS, len(models)))
print("Model name\t\t\tlikes\tdownloads\tlibrary\ttask\tdatasets")
for model in models[:N_MODELS]:
    print("{}\t{}\t{}\t{}\t{}\t{}".format(model.modelId, model.likes, model.downloads, model.task, getattr(model,"library_name","None"),
                                          model.tags, model.card_data.datasets))
