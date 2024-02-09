from evaluate import load, evaluator
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor import INCQuantizer
from analysis.utils import *
from transformers import pipeline


model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
dataset = load_dataset("tweet_eval", 'sentiment', split="test").select(range(64))
processor = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

pipe = pipeline('text-classification', model=model, tokenizer=processor)
# Set up quantization configuration and the maximum number of trials to 10
tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)

# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",  # Change as wished
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)


def eval_func(model):
    pipe.model = model
    # SequenceClassification is actually "text-classification"
    task_evaluator = evaluator('text-classification')
    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        #tokenizer=processor,
        data=dataset,
        metric=load("accuracy"),
        # input_column=dataset.column_names[0],
        # label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]


def run_optimization(save_dir):

    quantizer = INCQuantizer.from_pretrained(model=model,
                                             eval_fn=eval_func
                                             )
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)



run_optimization('./optim_config')
#if __name__ == "__main__":
#    model_data = get_model_data_from_line(sys.argv[1])
#    dataset = get_split_dataset(model_data, train_size=0.5, seed=SEED, split='test')
    #processor = AutoTokenizer.from_pretrained(model_data['model_name'], model_max_length=512)
    #processor = get_processor_from_category(model_data[̈́'category'], model_data[̈́'model_name'])
#    run_optimization(sys.argv[2])
