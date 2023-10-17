import timm
from datasets import load_dataset
from neural_compressor import PostTrainingQuantConfig
from neural_compressor.data import DataLoader, Datasets
from neural_compressor.quantization import fit

dataset = load_dataset("MMInstruction/M3IT", split="test")  # Selecting only the test split for the dataset

# Use the prepare_tf_dataset method from 🤗 Transformers to prepare the dataset to be compatible with TensorFlow,
# and ready to train/fine-tune a model, as it wraps a HuggingFace Dataset as a tf.data.Dataset with collation and
# batching, so one can pass it directly to Keras methods like fit() without further modification.



dataloader = DataLoader(framework="tensorflow", dataset=dataset)    #Keep tensorflow as the main framework

model = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True)

tf_dataset = model.prepare_tf_dataset(
    dataset,
    batch_size=4,
    shuffle=True
)


q_model = fit(
    model=model,
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader
)