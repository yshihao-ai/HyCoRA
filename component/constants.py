from hyper_model.chatglm2.configuration_chatglm import HLChatGLMConfig
from hyper_model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from hyper_model.glm2_hypernetwork.hyper_network import Chatglm2HyperNetwork
from hyper_model.llama import LlamaForCausalLM
from hyper_model.llama.configuration_llama import HLLlamaConfig
from hyper_model.llama_hypernetwork.hyper_network import Llama2HyperNetwork

from hyper_model.qwen2 import Qwen2ForCausalLM
from hyper_model.qwen2.configuration_qwen2 import HLQwen2Config
from hyper_model.qwen2_hypernetwork.hyper_network import Qwen2HyperNetwork


HYPER_MODEL_CONFIG = {
    "chatglm2": HLChatGLMConfig,
    "llama": HLLlamaConfig,
    "qwen2": HLQwen2Config
}

HYPER_MODEL = {
    "chatglm2": ChatGLMForConditionalGeneration,
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM
}

HYPER_NETWORK = {
    "chatglm2": Chatglm2HyperNetwork,
    "llama": Llama2HyperNetwork,
    "qwen2": Qwen2HyperNetwork   
}

LAYERNORM_NAMES = {"norm", "ln"}