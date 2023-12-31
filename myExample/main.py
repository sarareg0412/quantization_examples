from datasets import load_dataset, Dataset, Features
from transformers import  ViTForImageClassification, AutoImageProcessor, DefaultDataCollator
from torch.utils.data import DataLoader
from neural_compressor.data import DataLoader as DLNC

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.quantization import fit

# 1.0 Download the beans test dataset
dataset = load_dataset("beans", split="test")

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 1.1 TBD Add proper data Transformation step
# Converting Image() to cacheable PIL.Image object
def transforms(examples):
    examples["pixel_values"] = [image.convert("RGB") for image in examples["image"]]
    return examples


dataset = dataset.map(transforms, remove_columns=["image"], batched=True)

# 1.3 Before we fine-tune the model we need to convert our Hugging Face datasets Dataset into a tf.data.Dataset:
# Data collator that will dynamically pad the inputs received, as well as the labels.

# Setting the dataset's features
#features = Features({"image": dataset.features["image"], "label": dataset.features["labels"]})
#ds = Dataset.from_dict({"image": dataset["image"], "label": dataset["labels"]}).with_format("torch")
#dataloader = DLNC(dataset=ds, framework="pytorch", batch_size=4, num_workers=4)


# 2 Load model directly
model = ViTForImageClassification.from_pretrained("nateraw/vit-base-beans")

# You can parallelize data loading with the num_workers argument of a PyTorch DataLoader and get a higher throughput.
dataloader = DLNC(framework="pytorch", dataset=dataset)

model.eval()
print("Starting model quantization")

q_model = fit(
    model=model,
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader
)

print("Saving quantized model")
# q_model.save("./output")
