from datasets import load_dataset, Dataset, Features
from transformers import DefaultDataCollator, TFViTForImageClassification
from torch.utils.data import DataLoader
from neural_compressor.data import DataLoader as DLNC

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.quantization import fit

# 1.0 Download the beans test dataset
dataset = load_dataset("beans", split="test")

#image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 1.1 TBD Add proper data Transformation step
# Converting Image() to cacheable PIL.Image object
def transforms(examples):
    examples["pixel_values"] = [image.convert("RGB") for image in examples["image"]]
    return examples


dataset = dataset.map(transforms, remove_columns=["image"], batched=True)

# 1.3 Before we fine-tune the model we need to convert our Hugging Face datasets Dataset into a tf.data.Dataset:
# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator = DefaultDataCollator(return_tensors="tf")
tf_dataset = dataset.to_tf_dataset(
   columns=['pixel_values'],
   label_cols=["labels"],
   shuffle=True,
   batch_size=4,
   collate_fn=data_collator
)

# 2 Load tensorflow model directly
model = TFViTForImageClassification.from_pretrained("nateraw/vit-base-beans", from_pt=True)

# You can parallelize data loading with the num_workers argument of a PyTorch DataLoader and get a higher throughput.
dataloader = DLNC(framework="tensorflow", dataset=tf_dataset)

#model.eval()
print("Starting model quantization")

q_model = fit(
    model=model,
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader
)

print("Saving quantized model")
q_model.save("./output")
