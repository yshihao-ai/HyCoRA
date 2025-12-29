from typing import Any, Dict, List
import torch
from loguru import logger


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            if self.tokenizer.padding_side == 'left':
                input_ids = [self.pad_token_id] * padding_len + input_ids
                attention_mask = [0] * padding_len + attention_mask
                target_mask = [0] * padding_len + target_mask
            else:
                input_ids = input_ids + [self.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                target_mask = target_mask + [0] * padding_len
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs


class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [x['input_ids'] for x in batch if x['input_ids'] is not None]
        lengths = [len(x) for x in batch]
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for x in batch:
            input_ids = x
            attention_mask = [1] * len(input_ids)

            padding_len = batch_max_len - len(input_ids)
            labels = input_ids + [-100] * padding_len
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }
        return inputs


class HyperDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch, role_ids_batch = [], [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            role_ids = x['role_ids']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            if self.tokenizer.padding_side == 'left':
                input_ids = [self.pad_token_id] * padding_len + input_ids
                attention_mask = [0] * padding_len + attention_mask
                target_mask = [0] * padding_len + target_mask
            else:
                input_ids = input_ids + [self.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                target_mask = target_mask + [0] * padding_len
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
            role_ids_batch.append(role_ids)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        role_ids_batch = torch.tensor(role_ids_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'role_ids': role_ids_batch,
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs