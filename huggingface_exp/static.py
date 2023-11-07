from functools import partial
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel import INCQuantizer

# Applying static quantization
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# The directory where the quantized model will be saved
save_dir = "static_quantization"

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(approach="static")
quantizer = INCQuantizer.from_pretrained(model)

# Generate the calibration dataset needed for the calibration step
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
    num_samples=100,
    dataset_split="train",
)
#quantizer = INCQuantizer.from_pretrained(model)
# Apply static quantization and save the resulting model
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    save_directory=save_dir,
)