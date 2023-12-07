from transformers import AutoModel, AutoModelForImageClassification, AutoImageProcessor, pipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

model_name = "nateraw/vit-base-beans"
save_directory = "./beans_exp/model_inc"
# With the AutoModel Class it doesn't work. We need to manually select the class from the file's task
model = AutoModelForImageClassification.from_pretrained(save_directory)
processor = AutoImageProcessor.from_pretrained(model_name)
# Using
classifier = pipeline("image-classification", model=model, image_processor=processor)

dataset = load_dataset("beans", split="test[:10]")

for out in classifier(KeyDataset(dataset, "image")):
    print(out)