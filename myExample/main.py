import numpy as np
from transformers import AutoFeatureExtractor
import timm
from datasets import load_dataset
from neural_compressor import PostTrainingQuantConfig
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit

#model = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True)
model = timm.create_model("hf_hub:Bingsu/timm-mobilevitv2_050-beans", pretrained=True)

dataset = load_dataset("beans", split="test")  # Selecting only the test split of the dataset
tf_ds = dataset.to_tf_dataset(
            columns=["image"],
            label_cols=["labels"],
            batch_size=2,
            shuffle=False
            )

#Data augmentation step
#feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

#def transforms(examples):
#    examples["pixel_values"] = [
#        np.array(image)["image"] for image in examples["image"]
#    ]
#    return examples

dataloader = DataLoader(framework='tensorflow', dataset=tf_ds)

print("Starting model quantization")

q_model = fit(
    model=model,
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader
)

print("Saving quantized model")
q_model.save("./output")
