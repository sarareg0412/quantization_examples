import sys

import pandas as pd
from evaluate import evaluator

from utils import *

from optimum.intel import INCModelForTokenClassification

def compare_models_HF(line, metric):
    model_data = get_model_data_from_line(line)
    data = (load_dataset(model_data["dataset"], model_data["dataset_config_name"], split="test"))
    data = (data.train_test_split(train_size=TEST_DATA_PERCENT, seed=SEED)["train"])

    task_evaluator = evaluator(model_data['task'])
    q_model = INCModelForTokenClassification.from_pretrained(get_quantized_model_path(model_data["category"], model_data["model_name"]))
    q_model.task = model_data['task']
    #q_model.task = model_data['task']
    nq_model = model_data['model_name']
    models = [ q_model]
    results = []
    for model in models:
        results.append(
            task_evaluator.compute(
                model_or_pipeline=model, data=data, metric=metric,
                tokenizer=get_processor_from_category(model_data['category'], model_data['model_name'])
            )
        )
        print('OK')
    df = pd.DataFrame(results)
    df[["overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
    print(df.to_markdown)
    df.to_csv(f'./INCModelForTokenClassification/{format_name(model_data["model_name"])}/evaluation_output.csv')

if __name__ == "__main__":
    compare_models_HF(sys.argv[1], sys.argv[2])
