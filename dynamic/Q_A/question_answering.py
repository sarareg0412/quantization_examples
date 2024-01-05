import evaluate
from optimum.intel import INCQuantizer
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from neural_compressor.config import AccuracyCriterion, TuningCriterion, PostTrainingQuantConfig

model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eval_dataset = load_dataset("squad", split="validation").select(range(64))
task_evaluator = evaluate.evaluator("question-answering")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def eval_fn(model):
    qa_pipeline.model = model
    metrics = task_evaluator.compute(model_or_pipeline=qa_pipeline, data=eval_dataset, metric="squad")
    return metrics["f1"]

# Set the accepted accuracy loss to 5%
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Set the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
quantization_config = PostTrainingQuantConfig(
    approach="dynamic", accuracy_criterion=accuracy_criterion, tuning_criterion=tuning_criterion
)
quantizer = INCQuantizer.from_pretrained(model, eval_fn=eval_fn)
quantizer.quantize(quantization_config=quantization_config, save_directory="dynamic_quantization")