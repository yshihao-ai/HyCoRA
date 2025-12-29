import argparse
import json
import os
from dataclasses import dataclass, field
from os.path import join
import pandas as pd
import torch
import random
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(grandparent_dir)
from component.utils import ModelUtils
from component.constants import HYPER_MODEL_CONFIG
from hyper_model.chatglm2.configuration_chatglm import HLChatGLMConfig, ChatGLMConfig
from script.evaluate.eval_rouge import evaluate_rouge
from script.evaluate.eval_bleu import evaluate_bleu

@dataclass
class EvaluateArguments:
    dataset_path: str = field(metadata={"help": "Test dataset path"})
    role_desc_file: str = field(metadata={"help": "role description path"})
    output_dir: str = field(metadata={"help": "output path for evaluation results"})
    model_name_or_path: str = field(metadata={"help": "backbone model path"})
    adapter_name_or_path: str = field(default=None, metadata={"help": "adapter weights path"})
    train_mode: str = field(default="lora", metadata={"help": "Training strategy: [full, qlora, lora, hyperlora]"})
    language: str = field(default='zh',  metadata={"help": "Language: [zh, en]"})

    do_sample: bool = field(default=False, metadata={"help": "Whether to sample"})
    temperature: float = field(default=1.0, metadata={"help": "temperature"})
    top_p: float = field(default=1.0, metadata={"help": "top_p"})
    top_k: int = field(default=50, metadata={"help": "top_k"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition_penalty"})
    max_new_tokens: int = field(default=100, metadata={"help": "Maximum number of generated tokens"})
    batch_size: int = field(default=96, metadata={"help": "Batch size during inference"})
    roles_num: int = field(default=5, metadata={"help": "Number of roles"})
    roles_emb_dim: int = field(default=128, metadata={"help": "Role embedding dimension"})
    layers_num: int = field(default=112, metadata={"help": "Number of layers in the large model"})
    layers_emb_dim: int = field(default=128, metadata={"help": "Layer embedding dimension"})
    residual_blocks_num: int = field(default=2, metadata={"help": "Number of residual blocks in the hyper-network"})
    hyper_hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension of the hyper-network"})
    rank_dim: int = field(default=8, metadata={"help": "Rank of HyperLoRA"})
    alpha: int = field(default=32, metadata={"help": "LoRA scaling factor (alpha)"})
    layernorm_input: bool = field(default=True, metadata={"help": "Whether to apply layer normalization to the hyper-network input"})
    layernorm_output: bool = field(default=True, metadata={"help": "Whether to apply layer normalization to the hypernetwork output"})
    dropout: float = field(default=0.0, metadata={"help": "Whether to apply dropout in the hyper-network"})
    
    
    def __str__(self):
        class_name = self.__class__.__name__
        attributes = ",\n\t".join([f"{attr}={getattr(self, attr)}" for attr in dir(self) if not attr.startswith("__")])
        return f'{class_name}' + '{' + '\n\t' + f'{attributes}' + '\n}'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

def load_eval_dataset(dataset_path):
    logger.info('Loading evaluate data: {}'.format(dataset_path))
    df = pd.read_json(dataset_path, lines=True).to_dict(orient="records")
    logger.info('evaluate data lengths: {}'.format(len(df)))
    return df


def load_role_to_id_dict(role_desc_file):
    logger.info('Loading role\'s description: {}'.format(role_desc_file))
    role_desc = pd.read_json(role_desc_file, typ="series").to_dict()
    role_to_id = dict()
    for idx, role in enumerate(role_desc.keys()):
        role_to_id[role] = idx
    logger.info("There are {} role in dataset".format(len(role_desc.keys())))
    return role_to_id


def load_model_and_tokenizer(args, device=torch.device('cuda')):
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = args.adapter_name_or_path
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    if args.train_mode != "hyperlora":
        model = ModelUtils.load_model(model_name_or_path, adapter_name_or_path=adapter_name_or_path).eval()
    else:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if 'chatglm2' in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['chatglm2']
        elif 'llama' in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['llama']
        elif 'qwen2' in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['qwen2']
        else:
            raise ValueError(f'Unrecognized hyper model config: {args.model_name_or_path}')
        hl_config = hyper_config(
            roles_num=args.roles_num,
            roles_emb_dim=args.roles_emb_dim,
            layers_num=args.layers_num,
            layers_emb_dim=args.layers_emb_dim,
            residual_blocks_num=args.residual_blocks_num,
            hyper_hidden_dim=args.hyper_hidden_dim,
            rank_dim=args.rank_dim,
            alpha=args.alpha,
            layernorm_input=args.layernorm_input,
            layernorm_output=args.layernorm_output,
            dropout=args.dropout,
            **config.to_dict()
        )
        model = ModelUtils.load_hyper_model(model_name_or_path, adapter_name_or_path, hl_config).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == 'llama' else True
    )
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    
    tokenizer.padding_side = 'left'
    model = model.to(device)

    return model, tokenizer


def generate(model, tokenizer, args, question, role_ids):
    model_name = args.model_name_or_path.strip('/').split('/')[-1]
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if 'chatglm' in model_name.lower():
        inputs = tokenizer(question, return_tensors="pt", padding=True)
        if args.train_mode == "hyperlora":
            inputs["role_ids"] = torch.tensor(role_ids, dtype=torch.long)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        outputs = [outputs.tolist()[i][len(inputs["input_ids"][i]):] for i in range(len(question))]
        responses = tokenizer.batch_decode(outputs)
        responses = [model.process_response(r) for r in responses]
    elif 'llama' in model_name.lower():
        inputs = tokenizer(question, add_special_tokens=False, return_tensors="pt", padding=True)
        if args.train_mode == "hyperlora":
            inputs["role_ids"] = torch.tensor(role_ids, dtype=torch.long)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        outputs = [outputs.tolist()[i][len(inputs["input_ids"][i]):] for i in range(len(question))]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    elif 'qwen2' in model_name.lower():
        inputs = tokenizer(question, add_special_tokens=False, return_tensors="pt", padding=True)
        if args.train_mode == "hyperlora":
            inputs["role_ids"] = torch.tensor(role_ids, dtype=torch.long)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        outputs = [outputs.tolist()[i][len(inputs["input_ids"][i]):] for i in range(len(question))]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        responses = ["" for i in range(len(question))]
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_args_file", type=str, default='eval_args/sft/lora/chatglm2-7b-sft-lora.json', help="")
    args = parser.parse_args()
    eval_args_file = args.eval_args_file
    parser = HfArgumentParser((EvaluateArguments,))
    args, = parser.parse_json_file(json_file=eval_args_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.add(join(args.output_dir, 'eval.log'))
    logger.info("eval_args:\n{}".format(args))

    with open(eval_args_file, "r") as f:
        eval_args = json.load(f)

    with open(join(args.output_dir, 'eval_args.json'), "w") as f:
        json.dump(eval_args, f, indent=4)

    model, tokenizer = load_model_and_tokenizer(args)

    df = load_eval_dataset(args.dataset_path)
    role_to_id = load_role_to_id_dict(args.role_desc_file)
 
    result, reference = [], []
    batch_size = args.batch_size
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[i: i + min(len(df), batch_size)]
        batch_references = []
        batch_queries = []
        batch_roles = []
        for data in batch:
            instruction, query, role = data["instruction"], data["input"], data["role"]
            if 'chatglm2' in args.model_name_or_path.lower():
                if args.train_mode == 'hyperlora':
                    query = query.strip()
                else:
                    query = '\n'.join([instruction.strip(), query.strip()])
                query = '[Round 1]\n\n问：{}\n\n答：'.format(query)
            elif 'llama' in args.model_name_or_path.lower():
                if args.train_mode == 'hyperlora':
                    query ='[INST]' + query  + '[/INST]'
                else:
                    query = '<<SYS>>\n' + instruction + '\n<</SYS>>\n\n' + '[INST]' + query  + '[/INST]'
            elif 'qwen2.5' in args.model_name_or_path.lower():
                if args.train_mode == 'hyperlora':
                    query = 'User: {}\nAssistant: '.format(query)
                else:
                    system = 'System: {}\n'.format(instruction)
                    query = system + 'User: {}\nAssistant: '.format(query)
            elif 'qwen' in args.model_name_or_path.lower():
                if args.train_mode == 'hyperlora':
                    query = 'User: {}\nAssistant: '.format(query)
                else:
                    system = 'System: {}\n'.format(instruction)
                    query = system + 'User: {}\nAssistant: '.format(query)
            batch_queries.append(query.strip())
            batch_roles.append(role_to_id[role])
            batch_references.append(data['output'])

        batch_responses = generate(model, tokenizer, args, batch_queries, batch_roles)
        batch_responses = [res.strip() for res in batch_responses]
        result.extend(batch_responses)
        reference.extend(batch_references)

    with open(join(args.output_dir, 'predict_generated.txt'), "w", encoding='utf-8') as f:
        for res in result:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    metrics = evaluate_rouge(result, reference, args.language)
    metrics.update(evaluate_bleu(result, reference, args.language))

    logger.info("eval_result:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

if __name__ == '__main__':
    main()
