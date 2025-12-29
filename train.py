import argparse
from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from component.collator import PretrainCollator, SFTDataCollator, HyperDataCollator
from component.argument import CustomizedArguments
from component.template import template_dict
from component.dataset import (
    UnifiedSFTDataset,
    HyperUnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset, 
    HyperChatGLM2SFTDataset
)
from component.checkpointing import prepare_model_for_training
from component.constants import HYPER_MODEL_CONFIG, HYPER_MODEL, HYPER_NETWORK
from component.trainer import HyperTrainer
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)
import importlib

from hyper_model.chatglm2.configuration_chatglm import ChatGLMConfig, HLChatGLMConfig
from hyper_model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration

if importlib.util.find_spec('unsloth') is not None:
    from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from trl import DPOTrainer, get_kbit_device_map
import torch.nn as nn

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/sft/qlora/qwen-7b-sft-qlora.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    set_seed(training_args.seed)

    assert args.task_type in ['pretrain', 'sft', 'dpo'], "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora', 'hyperlora'], "task_type should be in ['full', 'lora', 'qlora', 'hyperlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"
    return args, training_args


def find_all_linear_names(model, train_mode, target_modules):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if target_modules != 'all':
        is_subset = set(target_modules).issubset(lora_module_names)
        if is_subset:
            lora_module_names = target_modules
        else:
            raise ValueError(f"target_modules {target_modules} not in {lora_module_names}")

    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def load_pretrain_dataset(training_args, args, tokenizer):
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"

    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    logger.info(f"padding_side of tokenizer: {tokenizer.padding_side}")
    logger.info(f"eos_token_id of tokenizer: {tokenizer.eos_token_id}")
    logger.info(f"pad_token of tokenizer: {tokenizer.pad_token}")

    return tokenizer


def load_unsloth_model(args, training_args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        trust_remote_code=True,
        load_in_4bit=True if args.train_mode == 'qlora' else False,
    )
    if args.train_mode in ['lora', 'qlora']:
        logger.info('Initializing PEFT Model...')
        target_modules = find_all_linear_names(model, args.train_mode)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=training_args.seed,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f'target_modules: {target_modules}')
    return {
        'model': model,
        'ref_model': None,
        'peft_config': None
    }


def load_model(args, training_args, dataset_len):
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')

    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16

    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if args.train_mode == 'hyperlora':
        del model_kwargs['use_cache']
        if "chatglm2" in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['chatglm2']
            hyper_model = HYPER_MODEL['chatglm2']
            hypernetwork = HYPER_NETWORK['chatglm2']
        elif 'llama' in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['llama']
            hyper_model = HYPER_MODEL['llama']
            hypernetwork = HYPER_NETWORK['llama']
        elif 'qwen2' in args.model_name_or_path.lower():
            hyper_config = HYPER_MODEL_CONFIG['qwen2']
            hyper_model = HYPER_MODEL['qwen2']
            hypernetwork = HYPER_NETWORK['qwen2']
        else:
            raise ValueError(f'Unrecognized hyper model: {args.model_name_or_path}')

        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        
        if hasattr(config, 'pad_token_id') and config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

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
        hyper_network = hypernetwork(hl_config)
        if training_args.gradient_checkpointing:
            hl_config.use_cache = False
        model_kwargs['config'] = hl_config
        model = hyper_model.from_pretrained(args.model_name_or_path, **model_kwargs)
        batch_size = (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        steps_per_epoch = (dataset_len + batch_size - 1) // batch_size
        model.hyperlora_init(hyper_network, steps_per_epoch, training_args.num_train_epochs, training_args.gradient_accumulation_steps)
        model.mark_only_hyper_network_as_trainable()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True

    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    if training_args.gradient_checkpointing:
        prepare_model_for_training(model, args)

    if args.train_mode == 'full' or args.train_mode == 'hyperlora':
        peft_config = None
    else:
        target_modules = find_all_linear_names(model, args.train_mode, args.target_modules)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.train_mode == 'full' else None
    else:
        ref_model = None

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total trainable params: %.2fM" % (total_trainable / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    
    template = template_dict[args.template_name]
    
    if 'chatglm2' in args.model_name_or_path.lower():
        if args.train_mode == 'hyperlora':
            logger.info('Loading data with HyperChatGLM2SFTDataset')
            train_dataset = HyperChatGLM2SFTDataset(args.role_desc_file,args.train_file, tokenizer, args.max_seq_length, template)
        else:
            logger.info('Loading data with ChatGLM2SFTDataset')
            train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    else:
        if args.train_mode == 'hyperlora':
            logger.info('Loading data with HyperUnifiedSFTDataset')
            train_dataset = HyperUnifiedSFTDataset(args.role_desc_file,args.train_file, tokenizer, args.max_seq_length, template)
        else:
            logger.info('Loading data with UnifiedSFTDataset')
            train_dataset = UnifiedSFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    train_dataset = UnifiedDPODataset(args.train_file, tokenizer, args.max_seq_length, args.max_prompt_length, template)
    return train_dataset


def init_components(args, training_args):
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    tokenizer = load_tokenizer(args)

    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        if args.train_mode == 'hyperlora':
            data_collator = HyperDataCollator(tokenizer, args.max_seq_length)
        else:
            data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with dpo task')
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None

    if args.use_unsloth:
        components = load_unsloth_model(args, training_args)
    else:
        components = load_model(args, training_args, len(train_dataset))
    
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
            print(p.dtype)
    
    if args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    else:
        if args.train_mode=='hyperlora':
            trainer = HyperTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        else:
             trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )           
    return trainer


def main():
    args, training_args = setup_everything()
    trainer = init_components(args, training_args)
    logger.info("*** starting training ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
