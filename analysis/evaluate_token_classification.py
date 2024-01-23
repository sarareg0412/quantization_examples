import sys

import pandas as pd
from evaluate import evaluator

from utils import *

task_evaluator = evaluator("token-classification")

def compare_token_classification_models(line):
    model_data = get_model_data_from_line(line)
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = (data.train_test_split(train_size=TEST_DATA_PERCENT, seed=SEED)["train"])


    q_model = get_quantized_model_path(model_data["category"], model_data["model_name"])
    nq_model = model_data['model_name']
    models = [nq_model, q_model]
    results = []
    for model in models:
        results.append(
            task_evaluator.compute(
                model_or_pipeline=model, data=data, metric="seqeval"
            )
        )
    df = pd.DataFrame(results)
    df[["overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
    print(df)

if __name__ == "__main__":
    compare_token_classification_models(sys.argv[1])