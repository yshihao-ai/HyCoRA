import json
import random
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset


class UnifiedSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system
        self.max_seq_length = max_seq_length

        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)

        input_ids, target_mask = [], []

        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.system
            if system is not None:
                system_text = self.system_format.format(content=system, stop_token=self.tokenizer.eos_token)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        filtered_input_ids = [input_ids[i] for i in range(len(input_ids)) if target_mask[i] == 1]

        decoded_filtered_input = self.tokenizer.decode(filtered_input_ids, skip_special_tokens=False)



        return inputs

class HyperUnifiedSFTDataset(Dataset):
    def __init__(self,role_desc_file, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system
        self.max_seq_length = max_seq_length

        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        
        self.role_file = role_desc_file
        logger.info('Loading role\'s data: {}'.format(role_desc_file))
        role_desc = pd.read_json(role_desc_file, typ="series").to_dict()
        role_to_id = dict()
        for idx, role in enumerate(role_desc.keys()):
            role_to_id[role] = idx
        logger.info("There are {} role in dataset".format(len(role_desc.keys())))
        self.role_desc = role_desc
        self.role_to_id = role_to_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []
        
        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()
            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'role_ids': self.role_to_id[data['role']],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

class ChatGLM2SFTDataset(UnifiedSFTDataset):
    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)

        input_ids = self.tokenizer.get_prefix_tokens()
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, idx=i + 1)
            assistant = self.assistant_format.format(content=assistant)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class HyperChatGLM2SFTDataset(UnifiedSFTDataset):
    def __init__(self, role_desc_file, *args):
        super(HyperChatGLM2SFTDataset, self).__init__(*args)
        self.role_file = role_desc_file

        logger.info('Loading role\'s data: {}'.format(role_desc_file))
        role_desc = pd.read_json(role_desc_file, typ="series").to_dict()
        role_to_id = dict()
        for idx, role in enumerate(role_desc.keys()):
            role_to_id[role] = idx
        logger.info("There are {} role in dataset".format(len(role_desc.keys())))
        self.role_desc = role_desc
        self.role_to_id = role_to_id

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)

        input_ids = self.tokenizer.get_prefix_tokens()
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, idx=i + 1)
            assistant = self.assistant_format.format(content=assistant)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'role_ids': self.role_to_id[data['role']],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM3SFTDataset(UnifiedSFTDataset):

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        system = data['system'].strip() if 'system' in data.keys() else self.system
        input_ids = self.tokenizer.get_prefix_tokens() + \
                    [self.tokenizer.get_command(f"<|system|>")] + \
                    self.tokenizer.encode(system, add_special_tokens=False)
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            input_tokens = [self.tokenizer.get_command(f"<|user|>")] + \
                           self.tokenizer.encode(human, add_special_tokens=False) + \
                           [self.tokenizer.get_command(f"<|assistant|>")]
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class UnifiedDPODataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, max_prompt_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def build_prompt_input_ids(self, system, history):
        if self.template_name in ['chatglm2', 'chatglm3']:
            prompt_input_ids = self.tokenizer.get_prefix_tokens()
        else:
            prompt_input_ids = []

        if self.system_format is not None:
            system = system if system is not None else self.system
            if system is not None:
                if self.template_name == 'chatglm3':
                    prompt_input_ids += [self.tokenizer.get_command(f"<|system|>")] + self.tokenizer.encode(system,
                                                                                                            add_special_tokens=False)
                else:
                    system_text = self.system_format.format(content=system)
                    prompt_input_ids += self.tokenizer.encode(system_text, add_special_tokens=False)

        for i, conv in enumerate(history):
            role = conv['role'].strip()
            content = conv['content'].strip()

            assert role != 'system', 'there should not be more than one system information'
            if role == 'user':
                if self.template_name == 'chatglm2':
                    human = self.user_format.format(content=content, idx=i // 2 + 1)
                    input_ids = self.tokenizer.encode(human, add_special_tokens=False)
                elif self.template_name == 'chatglm3':
                    input_ids = [self.tokenizer.get_command(f"<|user|>")] + \
                                self.tokenizer.encode(content, add_special_tokens=False) + \
                                [self.tokenizer.get_command(f"<|assistant|>")]
                else:
                    human = self.user_format.format(content=content, stop_token=self.tokenizer.eos_token)
                    input_ids = self.tokenizer.encode(human, add_special_tokens=False)
            elif role == 'assistant':
                if self.template_name in ['chatglm2', 'chatglm3']:
                    input_ids = self.tokenizer.encode(content, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                else:
                    assistant = self.assistant_format.format(content=content, stop_token=self.tokenizer.eos_token)
                    input_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
            else:
                raise Exception('role error')
            prompt_input_ids += input_ids

        return prompt_input_ids

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        chosen = data['chosen']
        rejected = data['rejected']
        assert len(chosen) == len(rejected)

        if chosen[0]['role'] == 'system':
            system = chosen[0]['content'].strip()
            history = chosen[1:-1]
            chosen, rejected = chosen[-1], rejected[-1]
        else:
            system = None
            history = chosen[:-1]
            chosen, rejected = chosen[-1], rejected[-1]

        prompt_input_ids = self.build_prompt_input_ids(system, history)

        if self.template_name in ['chatglm2', 'chatglm3']:
            chosen_input_ids = self.tokenizer.encode(chosen['content'], add_special_tokens=False) + [
                self.tokenizer.eos_token_id]
            rejected_input_ids = self.tokenizer.encode(rejected['content'], add_special_tokens=False) + [
                self.tokenizer.eos_token_id]
        else:
            chosen = self.assistant_format.format(content=chosen['content'], stop_token=self.tokenizer.eos_token)
            rejected = self.assistant_format.format(content=rejected['content'], stop_token=self.tokenizer.eos_token)

            chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
            rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1] * len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1] * len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1] * len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    def map(self, func, **kwargs):
        return self
