from optimum.intel.neural_compressor.quantization import INCModel
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation
)


class INCModelForImageClassification(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForImageClassification


class INCModelForSemanticSegmentation(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForSemanticSegmentation


def get_quantized_model_from_task(task, model_location):
    model = None
    match task:
        case "image-classification":
            model = INCModelForImageClassification.from_pretrained(model_location)
        case "image-segmentation":
            model = INCModelForSemanticSegmentation.from_pretrained(model_location)

    return model
