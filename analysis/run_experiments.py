from run_inference import *
from run_comparison import *

N_EXP = 10


def create_evaluation_csv(file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['category', 'model_name', 'metric', 'value', 'seed'])     # Create the header


def run_experiments(line):
    model_data = get_model_data_from_line(line)
    rng = np.random.default_rng(seed=SEED)
    seeds = list(rng.integers(low=0, high=100, size=N_EXP))
    print(seeds)
    file_name = f'./{model_data["category"]}/{format_name(model_data["model_name"])}/{format_name(model_data["model_name"])}_evaluation_results.csv'
    create_evaluation_csv(file_name)

    # Create the

    for seed in seeds:
        # Run inference for both models
        run_inference_from_line('True', line, seed)
        run_inference_from_line('False', line, seed)
        # Evaluate the models and add the results to a csv file
        result_dict = run_comparison(line, seed)
        content = format_result_dict(model_data, result_dict, seed)
        write_csv(file_name, content=content, mode='a')


def format_result_dict(model_data, result, seed):
    # Every line must follow this format: ['category', 'model_name', 'metric', 'value']
    lines = []
    for key in result:
        lines.append([model_data['category'], model_data['model_name'], key, result[key], seed])
    return lines



#if __name__ == "__main__":
#    run_experiments(sys.argv[1])