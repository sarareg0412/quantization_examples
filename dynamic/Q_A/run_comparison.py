
import evaluate
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from optimum.intel import INCModelForQuestionAnswering
from datasets import load_dataset
from tqdm import contrib

exact_match = evaluate.load("exact_match")

def run_comparison():
    data = load_dataset("squad", split="validation").select(range(64))
    data = data.train_test_split(train_size=0.5, seed=42)["test"]  # Use 50% of test dataset to run comparison

    # Get processor (image processor, tokenizer etc.)
    processor = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    # No need to retrieve the non quantized model as we only need its name to retrieve it from the hub
    # Retrieve quantized model by its configuration.
    q_model = INCModelForQuestionAnswering.from_pretrained("./config")
    # Setup non quantized and quantized model pipeline for inference
    nq_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer=processor)
    q_pipe = pipeline("question-answering", model=q_model, tokenizer=processor)
    # Initialize lists to store references and predictions for accuracy evaluation
    references = []
    nq_predictions = []
    q_predictions = []

    print(f"Evaluating Data for model distilbert-base-cased-distilled-squad")
    # Iterate through the test split
    for i,example in contrib.tenumerate(data):
        # Load object and label truth label from the dataset
        object = example["question"]  # Assume the object column name is the first one
        label = example["answers"]["text"][0]  # Assume the label column name is the last one

        # Infer the object label using the model
        nq_prediction = nq_pipe(question=object, context=example["context"])
        q_prediction = q_pipe(question=object, context=example["context"])
        # Since there might be multiple labels with multiple scores associated, we get the first one.
        nq_label = nq_prediction[0]['answer'] if isinstance(nq_prediction, list) \
            else nq_prediction['answer']
        q_label = q_prediction[0]['answer'] if isinstance(q_prediction, list) \
            else q_prediction['answer']

        # Append ground truth label and predicted label for accuracy evaluation
        print(f"A:{label} NQ:{nq_label} Q:{q_label}")
        references.append(label)
        nq_predictions.append(nq_label)  # Map the NQ predicted label using the q model's label2id attribute
        q_predictions.append(q_label)    # Map the Q predicted label using the q model's label2id attribute

    # Calculate accuracy using the loaded accuracy metric
    exact_match_score = exact_match.compute(predictions=q_predictions, references=nq_predictions)

    print(f"Exact match score is : {exact_match_score}")


run_comparison()
