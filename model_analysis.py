from huggingface_hub import HfApi, ModelFilter   # api to interact with the hub

hf_api = HfApi()
def add_ratio(model):
    model.ratio = model.likes/model.downloads
    return model

# We filter out models with 0 likes or downloads and turn everything into a list. We add the
# "ratio" parameter to the model, representing the ratio between #likes/#downloads
models = list(map(lambda x: add_ratio(x), filter(lambda x: x.likes > 10 and x.downloads > 10, hf_api.list_models(
    filter=ModelFilter(
        task="image-classification",
        library="transformers"
    ),
    sort="likes" and "downloads"
))))
#Sorting phase based on likes,downloads or ratio
models.sort(key=lambda x: x.likes, reverse=True)
print("Model name\t\t\tlikes\tdownloads\tratio\tlibrary\ttask")
for model in models[:10]:
    print("{}\t{}\t{}\t{:.3f}\t{}".format(model.modelId, model.likes, model.downloads, model.ratio, model.library_name))