from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
import evaluate

model_name = "juliensimon/distilbert-amazon-shoe-reviews"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eval_dataset = load_dataset("prashantgrao/amazon-shoe-reviews", split="test").select(range(300))


def eval_func(model):
    task_evaluator = evaluate.evaluator("text-classification")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_dataset,
        metric=evaluate.load("accuracy"),
        label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]


tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)
