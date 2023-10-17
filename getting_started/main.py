from neural_compressor.data import DataLoader, Datasets
from neural_compressor.config import PostTrainingQuantConfig

dataset = Datasets("tensorflow")["dummy"](shape=(1, 224, 224, 3))
dataloader = DataLoader(framework="tensorflow", dataset=dataset)

from neural_compressor.quantization import fit

q_model = fit(
    model="./mobilenet_v1_1.0_224_frozen.pb",
    conf=PostTrainingQuantConfig(),
    calib_dataloader=dataloader,
)