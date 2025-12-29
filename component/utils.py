import os

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
import safetensors
from peft import PeftModel
from component.constants import HYPER_MODEL_CONFIG, HYPER_MODEL, HYPER_NETWORK
from hyper_model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from hyper_model.llama.modeling_llama import LlamaForCausalLM

class ModelUtils(object):

    @classmethod
    def load_model(cls, model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float32,
            device_map='auto',
            quantization_config=quantization_config
        )

        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path)

        return model

    @classmethod
    def load_hyper_model(cls, model_name_or_path, adapter_name_or_path, config):
        model, hyper_model = None, None
        if 'chatglm2' in model_name_or_path.lower():
            hyper_model = HYPER_MODEL['chatglm2']
            hyper_network = HYPER_NETWORK['chatglm2']
        elif 'llama' in model_name_or_path.lower():
            hyper_model = HYPER_MODEL['llama']
            hyper_network = HYPER_NETWORK['llama']
        elif 'qwen2' in model_name_or_path.lower():
            hyper_model = HYPER_MODEL['qwen2']
            hyper_network = HYPER_NETWORK['qwen2']

        hyper_network = hyper_network(config)

        model = hyper_model.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            config=config,
        )
        model.hyperlora_init(hyper_network)

        if adapter_name_or_path is not None:
            pytorch_model_path = os.path.join(adapter_name_or_path, "pytorch_model.bin")
            safetensors_model_path = os.path.join(adapter_name_or_path, "model.safetensors")
            
            if os.path.exists(pytorch_model_path):
                hyper_state_dict = torch.load(pytorch_model_path)
            elif os.path.exists(safetensors_model_path):
                hyper_state_dict = safetensors.torch.load_file(safetensors_model_path)
            else:
                raise ValueError("No valid model file found: Neither pytorch_model.bin nor model.safetensors exists.")
            
            new_state_dict = model.state_dict()
            for k, v in hyper_state_dict.items():
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            raise ValueError("Loading hyper model: The 'adapter_name_or_path' cannot be None.")


        return model
