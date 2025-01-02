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
from model import BertForCL, BertConfig
from dataset import CLDataset
from tool import calc_loss, cosine_similarity, evaluate, plot
from tqdm import tqdm
import argparse
import random
jt.flags.use_cuda = jt.has_cuda
seed = 2048
jt.misc.set_global_seed(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# 读取配置文件
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
config_path = args.config

@dataclass
class ModelArguments:
    tokenizer_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The local dir of tokenizer"}
    )
    params_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the parameters"}
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the output parameters"}
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
            "help": "The maximum total input sequence length after tokenization. Sequences longer"
            "than this will be truncated"
        }
    )
    
@dataclass
class TrainingArguments:
    temperature: int = field(
        default=0.05,
        metadata={"help": "Temperature of Loss function"}
    )
    mode: str = field(
        default="unsupervised",
        metadata={"help": "The mode of the training. It must be \"supervised\" or \"unsupervised\"."}
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "batch size"}
    )
    epoch: int = field(
        default=1,
        metadata={"help": "epoch"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={"help": "learning_rate"}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to do evalution"}
    )
    print_step: int = field(
        default=10,
        metadata={"help": "Interval of printing results"}
    )
    eval_step: Optional[int] = field(
        default=50,
        metadata={"help": "Interval of doing evaluation and ploting"}
    )
    plot_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save plot image"}
    )
    negative_penalty: float = field(
        default=1.0,
        metadata={"help": "The penalty of negative example"}
    )
    dynamic_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use dynamic loss"}
    )
    def __post_init__(self):
        allowed_mode = ["supervised", "unsupervised"]
        if self.mode not in allowed_mode:
            raise ValueError("mode must be supervised or unsupervised")

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

with open(config_path, 'r', encoding='utf-8') as f:
    config_data = commentjson.load(f)
model_args, data_args, training_args = parser.parse_dict(config_data)

# 读取数据
if training_args.mode == "supervised":
    dataset = load_dataset("csv", data_files=data_args.dataset_path)
    dataset = dataset["train"].shuffle(seed=seed)
    # print(dataset['train'][0])
elif training_args.mode == "unsupervised":
    dataset = load_dataset("text", data_files=data_args.dataset_path)
    dataset = dataset["train"].shuffle(seed=seed)
    # print(dataset["train"][0])
else:
    raise ValueError("The mode must be \"supervised\" or \"unsupervised\".")

tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_dir, local_files_only=True)
# jt.display_memory_info()

training_dataset = CLDataset(dataset, tokenizer, data_args, training_args)
training_dataloader = DataLoader(training_dataset, batch_size=training_args.batch_size)

if training_args.mode == "supervised":
    bert_config = BertConfig(hidden_dropout_prob=0.1)
else:
    bert_config = BertConfig(hidden_dropout_prob=0.1)
    
model = BertForCL(bert_config).cuda()
params_dict = jt.load(model_args.params_path)
params_dict = {k.replace("LayerNorm.gamma", "LayerNorm.weight"): v for k, v in params_dict.items()}
params_dict = {k.replace("LayerNorm.beta", "LayerNorm.bias"): v for k, v in params_dict.items()}
model.load_state_dict(params_dict)
params_dict = None  # 不加这行代码会导致内存泄漏，没有明白原因，可能和全局变量引用导致计算图没有释放有关

optimizer = jt.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
model.train()

min_loss = float("inf")
best_sd = {}

correlation_list = []
loss_list = []
total_len = len(training_dataset) // training_args.batch_size
step = 0
max_correlation = 0.0
for i in range(training_args.epoch):
    for batch_idx, (y1, y2, y3) in enumerate(tqdm(training_dataloader, total=total_len)):
        input_ids1 = y1["input_ids"].squeeze(1)
        token_type_ids1 = y1["token_type_ids"].squeeze(1)
        attention_mask1 = y1["attention_mask"].squeeze(1)
        input_ids2 = y2["input_ids"].squeeze(1)
        token_type_ids2 = y2["token_type_ids"].squeeze(1)
        attention_mask2 = y2["attention_mask"].squeeze(1)

        if training_args.mode == "supervised":
            input_ids3 = y3["input_ids"].squeeze(1)
            token_type_ids3 = y3["token_type_ids"].squeeze(1)
            attention_mask3 = y3["attention_mask"].squeeze(1)

            _, z1 = model(input_ids1, token_type_ids1, attention_mask1)
            _, z2 = model(input_ids2, token_type_ids2, attention_mask2)
            _, z3 = model(input_ids3, token_type_ids3, attention_mask3)
            # z1, _ = model(input_ids1, token_type_ids1, attention_mask1)
            # z2, _ = model(input_ids2, token_type_ids2, attention_mask2)
            # z3, _ = model(input_ids3, token_type_ids3, attention_mask3)

            # input_ids4 = input_ids1.copy()
            # token_type_ids4 = token_type_ids1.copy()
            # attention_mask4 = attention_mask1.copy()
            # _, z4 = model(input_ids4, token_type_ids4, attention_mask4)

            # loss = calc_loss(training_args, z1, z4, z3)
            loss = calc_loss(training_args, z1, z2, z3)

            optimizer.step(loss)
            jt.sync_all()
            jt.gc()

            z1_z2_similarity = cosine_similarity(z1, z2).mean()
            z1_z3_similarity = cosine_similarity(z1, z3).mean()
            indices = jt.randperm(z3.shape[0])
            z3 = z3[indices]
            z1_z3_similarity = cosine_similarity(z1, z3).mean()

            loss = loss.detach().item()
            
        elif training_args.mode == "unsupervised":
            _, z1 = model(input_ids1, token_type_ids1, attention_mask1)
            _, z2 = model(input_ids2, token_type_ids2, attention_mask2)
            # z1, _ = model(input_ids1, token_type_ids1, attention_mask1)
            # z2, _ = model(input_ids2, token_type_ids2, attention_mask2)

            loss = calc_loss(training_args, z1, z2)
            optimizer.step(loss)
            jt.sync_all()
            jt.gc()
            
            loss = loss.detach().item()
        else:
            raise ValueError("The mode must be \"supervised\" or \"unsupervised\".")

        step += 1
        if step % training_args.print_step == 0:
            if training_args.mode == "supervised":
                print(f"Epoch {i}, step {step}, loss={loss},\
                     z1_z2_similarity={z1_z2_similarity.item()}, \
                     z1_z3_similarity={z1_z3_similarity.item()}, ")
            else:
                print(f"Epoch {i}, step {step}, loss={loss}")

        if step % training_args.eval_step == 0 and training_args.do_eval:
            spearman = evaluate(model, model_args.tokenizer_dir)
            print("correlation: %.2f" % spearman)
            loss_list.append(loss)
            correlation_list.append(spearman)
            if spearman > max_correlation:
                max_correlation = spearman
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

jt.save(best_sd, model_args.output_path)
plot(loss_list, correlation_list, training_args.plot_save_path)