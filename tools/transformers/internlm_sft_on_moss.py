import os
import copy

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset

class SFTDataset(Dataset):
    # https://github.com/OpenLMLab/MOSS/blob/main/finetune_moss.py
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.dataset[index]["input_ids"])
        no_loss_spans = copy.deepcopy(self.dataset[index]["no_loss_spans"])

        data = torch.tensor(data, dtype=torch.long)
        label = copy.deepcopy(data)

        for no_loss_span in no_loss_spans:
            label[no_loss_span[0] : no_loss_span[1]] = -100

        return data, label
    
def collate_fn(batch, tokenizer):
    batch_input_ids, batch_labels = [], []
    for input_ids, label in batch:
        batch_input_ids.append(input_ids)
        batch_labels.append(label)

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": (batch_input_ids == tokenizer.eos_token_id).long(),
        "labels": batch_labels
    }

def process(sample, tokenizer, max_len):
    chat = sample["plain_text"].split("<eoa>")[:-1]
    num_turns = sample["num_turns"]
    meta_instruction = sample["prefix"]

    # encode instruction
    instruction_ids = tokenizer.encode(meta_instruction)
    assert isinstance(instruction_ids, list), instruction_ids
    assert len(instruction_ids) > 0, len(instruction_ids)
    input_ids = copy.deepcopy(instruction_ids)
    # We do not calculate loss for instruction.
    no_loss_spans = [(0, len(instruction_ids))]

    for i in range(num_turns):
        # Collect dialogues
        cur_turn_ids = []
        cur_no_loss_spans = []
        # Add to cur_turn_ids
        cur_turn_ids.extend(tokenizer.encode(chat[i] + "<eoa>"))
        # if key == 'Tool Responses':
        #     # The format tokens (<|Results|>:...<eor>\n) should have losses. 
        #     cur_no_loss_spans.append((len(input_ids + cur_turn_ids) + 5, len(input_ids + cur_turn_ids + cur_ids) - 2))
        if len(input_ids + cur_turn_ids) > max_len:
            # Too long, break
            break
        # Extend input_ids
        input_ids.extend(cur_turn_ids)
        no_loss_spans.extend(cur_no_loss_spans)

    if len(input_ids) == len(instruction_ids):
        # No dialogue, return
        return {"input_ids": [], "no_loss_spans": []}
    else:
        return {"input_ids": input_ids, "no_loss_spans": no_loss_spans}


def load_data(save_dir, tokenizer, max_len, num=-1) -> HFDataset:
    if os.path.exists(save_dir):
        print(f"Loading moss-002-sft from {save_dir}")
    else:
        print(f"Loading moss-002-sft from datasets")
        moss_sft = load_dataset("fnlp/moss-002-sft-data", split="train")
        moss_sft = moss_sft.map(lambda x:process(x, tokenizer, max_len), num_proc=10)
        moss_sft = moss_sft.filter(lambda x:len(x["input_ids"]) != 0)
        moss_sft.save_to_disk(save_dir)

    moss_sft = HFDataset.load_from_disk(save_dir)
    if num != -1:
        moss_sft = moss_sft.select(range(num))
    print(
        f"Load successfully, total {len(moss_sft)} samples.")
    
    return moss_sft

def get_dataset(tokenizer, save_dir, max_len=1024, num=-1, test_size=0.1):
    moss_sft_data = load_data(save_dir, tokenizer, max_len, num)
    moss_sft_split = moss_sft_data.train_test_split(test_size=test_size)
    train_dataset = SFTDataset(moss_sft_split["train"])
    val_dataset = SFTDataset(moss_sft_split["test"])

    return train_dataset, val_dataset

