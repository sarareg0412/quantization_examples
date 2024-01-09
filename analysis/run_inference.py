from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from utils import *
from transformers import pipeline
from datasets import load_dataset
from tqdm import contrib
from evaluate import evaluator


def run_evaluation_from_line(quantized, line):
    model_data = get_model_data_from_line(line)
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = data.train_test_split(train_size=0.5, seed=SEED)["train"]  # Use 50% of test dataset to make inference

    quantized = True if (quantized == "True") else False
    model_path = get_quantized_model_path(model_data["category"], model_data["model_name"])
    # If we want to evaluate the NOT quantized model, we can just use the
    # HF model name as parameter to pass to the task_evaluator
    if not quantized:
        model_path = model_data["model_name"]

    model = get_model_from_library(model_data["library"], model_data["task"], model_path)
    processor = get_processor_from_category(model_data["category"], model_data["model_name"])

    compute_accuracy = False
    if compute_accuracy:
        task_evaluator = evaluator(model_data["task"])
        # Evaluate the model's accuracy if it's  (performs inference too)
        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=data,
            label_column=data.column_names[-1],  # We assume the last column is the labels one
            label_mapping=model.config.label2id,    # Extremely important
            feature_extractor=processor,
            tokenizer=processor
        )
        print(eval_results)
    else:
        pipe = pipeline(model_data["task"], model=model, image_processor=processor)

        print("PERFORMING INFERENCE")

        # map the dataset to a list of PIL.Image for input to the pipeline
        # pipe(inputs) should return a list of scores + labels

        for out in tqdm(pipe(KeyDataset(data, "image"))):
            print(out)
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

if __name__ == "__main__":
    #run_evaluation_from_line(sys.argv[1], sys.argv[2])
    run_evaluation_from_line("False","nateraw/vit-base-beans,10,2023,computer-vision,image-classification,transformers,beans,")