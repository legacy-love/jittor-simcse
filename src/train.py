import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from typing import Optional, List, Union, Dict, Tuple
import numpy as np
from dataclasses import dataclass, field
import commentjson

from transformers import (
    HfArgumentParser
)

from datasets import load_dataset
from transformers import BertTokenizerFast, AutoTokenizer
import jittor as jt
from jittor.dataset import Dataset, DataLoader
from model import BertForCL
from dataset import CLDataset

@dataclass
class ModelArguments:
    tokenizer_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The local dir of tokenizer"}
    )

@dataclass
class DataArguments:
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the dataset"}
    )
    max_seq_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences logonger"
            "than this will be truncated"
        }
    )
    
@dataclass
class TrainingArguments:
    mode: Optional[str] = field(
        default=None,
        metadata={"help": "The mode of the training. It must be \"supervised\" or \"unsupervised\"."}
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "batch size"}
    )
    epoch: int = field(
        default=10,
        metadata={"help": "epoch"}
    )
    def __post_init__(self):
        allowed_mode = ["supervised", "unsupervised"]
        if self.mode not in allowed_mode:
            raise ValueError("mode must be supervised or unsupervised")

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

config_path = "../config.jsonc"
with open(config_path, 'r', encoding='utf-8') as f:
    config_data = commentjson.load(f)
model_args, data_args, training_args = parser.parse_dict(config_data)

if training_args.mode == "supervised":
    dataset = load_dataset("csv", data_files=data_args.dataset_path)
elif training_args.mode == "unsupervised":
    dataset = load_dataset("text", data_files=data_args.dataset_path)
else:
    raise ValueError("The mode must be \"supervised\" or \"unsupervised\".")

tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_dir, local_files_only=True)


training_dataset = CLDataset(dataset, tokenizer, data_args, training_args)
training_dataloader = DataLoader(training_dataset, batch_size=training_args.batch_size)
