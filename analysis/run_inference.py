import csv
import sys

from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from utils import *
from transformers import pipeline
from datasets import load_dataset


def run_inference_from_line(quantized, line):
    model_data = get_model_data_from_line(line)
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = (data.train_test_split(train_size=0.05, seed=SEED)["train"])
            #.select(range(500)))  # Use 50% of test dataset to make inference

    quantized = True if (quantized == "True") else False
    model_path = get_quantized_model_path(model_data["category"], model_data["model_name"])
    # If we want to evaluate the NOT quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model_path = model_data["model_name"]

    model = get_model_from_library(model_data["library"], model_data["category"], model_path, quantized=quantized)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])
    output_file_name = (f"{model_data['category']}/{format_name(model_data['model_name'])}/"
                        f"{'' if quantized else 'N'}Q_output.csv")
    pipe = pipeline(model_data["task"], model=model, tokenizer=processor)

    print(f"PERFORMING INFERENCE on {'QUANTIZED ' if quantized else ' '}{model_data['model_name']} and "
          f"{model_data['dataset']}/{model_data['dataset_config_name']}")

    # map the dataset based on the category
    data = map_data(data,model_data["category"])

    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["label"])
        for out in tqdm(pipe(data)):
            # Since there might be multiple labels with multiple scores associated, we get the first one.
            prediction = get_prediction(out,model_data["category"], model.config.label2id)
            writer.writerow(prediction)
        print("Done")

        """
        # Iterate through the validation set or any other split
        for i, example in contrib.tenumerate(data):
            # Load object and label truth label from the dataset
            import pdb
            pdb.set_trace()

            object = example[data.column_names[1]]  # Assume the object column name is the first one
            label = example[data.column_names[-1]]  # Assume the label column name is the last one

            # Infer the object label using the model
            prediction = pipe(object)
            prediction = model(object)


            # Since there might be multiple labels with multiple scores associated, we get the first one.
            predicted_label = prediction[0]['label'] if isinstance(prediction, list) \
                                                                   else prediction['label']

            # Append ground truth label and predicted label for accuracy evaluation
            references.append(label)
            predictions.append(model.config.label2id[predicted_label])  # Map the predicted label using the model's label2id attribute
        """

def map_data(data, category):
    match category:
        case "INCModelForSequenceClassification":
            data = KeyDataset(data, "text")
        case "INCModelForQuestionAnswering":
            data = create_squad_examples(data)

    return data


def get_prediction(out, category, convert_fn):
    match category:
        case "INCModelForSequenceClassification":
            out = out[0]['label'] if isinstance(out['label'], list) else out['label']
            out = convert_fn(out)
        case "INCModelForQuestionAnswering":
            out = out[0]["answer"] if isinstance(out, list) else out["answer"]

    return [out]

if __name__ == "__main__":
    run_inference_from_line(sys.argv[1], sys.argv[2])
    #run_evaluation_from_line("True",
    #                         "cardiffnlp/twitter-roberta-base-sentiment-latest,277,22948384,INCModelForSequenceClassification,text-classification,transformers,tweet_eval,sentiment")
