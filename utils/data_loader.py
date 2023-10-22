import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random
import pandas as pd
import re
from sklearn.utils import shuffle


class MyDataset(Dataset):
    def __init__(self, datas, tokenizer, max_len):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.datas)

    def tokenize(self, text):
        input = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        input_ids = input['input_ids']
        mask = input['attention_mask']

        return input_ids, mask

    def __getitem__(self, index):
        df = self.datas.iloc[index]
        original_text, perturbed_text = df['original_text'], df['perturbed_text']
        original_ids, original_mask = self.tokenize(original_text)
        perturbed_ids,perturbed_mask = self.tokenize(perturbed_text)

        return {
            "original_ids": torch.tensor(original_ids, dtype=torch.long),
            "original_mask": torch.tensor(original_mask, dtype=torch.long),
            "perturbed_ids": torch.tensor(perturbed_ids, dtype=torch.long),
            "perturbed_mask":torch.tensor(perturbed_mask,dtype=torch.long)
        }

