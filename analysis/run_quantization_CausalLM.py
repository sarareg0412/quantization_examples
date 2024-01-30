from auto_gptq import AutoGPTQForCausalLM
from optimum.gptq import GPTQQuantizer
from transformers import AutoTokenizer
from utils import *
import sys


dataset = None


def run_quantization(save_dir):
    pretrained_model_name = model_data['model_name']
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    quantizer = GPTQQuantizer(bits=8, dataset=model_data['dataset'],
                              block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
    # The directory where the quantized model will be saved
    # Quantize and save the model
    quantized_model = quantizer.quantize_model(model, tokenizer)
    quantizer.save(quantized_model, save_dir)

if __name__ == "__main__":
    model_data = get_model_data_from_line(sys.argv[2])
    #dataset = get_dataset_from_name(model_data["dataset"], model_data["dataset_config_name"], QUANT_SPLIT_PERCENT)
    run_quantization(sys.argv[1])
