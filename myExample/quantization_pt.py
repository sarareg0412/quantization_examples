import numpy as np
from datasets import load_dataset
from torch._C._monitor import data_value_t
from transformers import  ViTForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader
from neural_compressor.data import DataLoader as DLNC

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.quantization import fit

# 1.0 Download the beans test dataset
dataset = load_dataset("beans", split="test")

# 1.1 TBD Add proper data Transformation step
def transforms(examples):
    examples["pixel_values"] = [np.asarray(image.resize((100,100))) for image in examples["image"]]
    return examples

dataset = dataset.map(transforms, batched=True)

# 1.2 Before we fine-tune the model we need to convert our Hugging Face datasets Dataset into our Dataset object:
# Creating own dataset object
class Dataset(object):
    def __init__(self):
        (test_images, test_labels) = dataset["pixel_values"], dataset["labels"]
        print("Converting dataset images to pixel values ")
        # Converting Image() to numpyArray and normalizing them
        self.test_images = np.asarray(test_images).astype(np.float32) / 255.0
        self.labels = test_labels

    def __getitem__(self, index):
        return self.test_images[index], self.labels[index]

    def __len__(self):
        return len(self.test_images)


ds = Dataset()

# 2 Load model directly
model = ViTForImageClassification.from_pretrained("nateraw/vit-base-beans")

# You can parallelize data loading with the num_workers argument of a PyTorch DataLoader and get a higher throughput.
dataloader = DLNC(framework="pytorch", dataset=ds)

#model.eval()
print("Starting model quantization")

q_model = fit(
    model=model,
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader
)

print("Saving quantized model")
# q_model.save("./output")
