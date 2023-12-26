
def eval_func(model, dataloader, metric):
    #  File "/home/sara/analysis/evaluation_functions.py", line 3, in eval_func
    # for input, label in dataloader:
    # ValueError: too many values to unpack (expected 2)
    for input, label in dataloader:
        output = model(input)
        metric.update(output, label)

    accuracy = metric.result()
    return accuracy