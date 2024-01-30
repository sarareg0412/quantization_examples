from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from utils import *
import sys


dataset = None


def run_quantization(save_dir):
    pretrained_model_name = model_data['model_name']
    quantize_config = BaseQuantizeConfig(bits=8, group_size=128)
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_name, quantize_config)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    examples = [
                tokenizer(
                            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
                                )
                ]
    print('Quantizing model')
    model.quantize(examples)
    print('Saving quantized model')
    model.save_quantized(save_dir, use_safetensors=False)

if __name__ == "__main__":
    model_data = get_model_data_from_line(sys.argv[2])
    #dataset = get_dataset_from_name(model_data["dataset"], model_data["dataset_config_name"], QUANT_SPLIT_PERCENT)
    run_quantization(sys.argv[1])
