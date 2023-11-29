from huggingface_hub import HfApi, ModelFilter, DatasetFilter  # api to interact with the hub
from utils import categories, category_dict

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
#import evaluate
#from optimum.intel.neural_compressor import INCQuantizer
hf_api = HfApi()

def get_dataset_name(model):
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
            return dataset


# Load model and dataset
model_name = "lambdalabs/sd-image-variations-diffusers"
model_api = list(hf_api.list_models(
            filter=ModelFilter(
                model_name= model_name
            ),
            limit=1,  # The limit on the number of models fetched.
            cardData=True
        ))[0]

model_api.dataset = get_dataset_name(model_api)

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers")
eval_dataset = load_dataset(model_api.dataset, split="test").select(range(300))

