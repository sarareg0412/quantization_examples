
def eval_func(model, dataloader, metric):
    for input, label in dataloader:
        output = model(input)
        metric.update(output, label)

    accuracy = metric.result()
    return accuracy