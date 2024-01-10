from optimum.intel.neural_compressor.quantization import INCModel
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation
)


class INCModelForImageClassification(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForImageClassification


class INCModelForSemanticSegmentation(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForSemanticSegmentation

