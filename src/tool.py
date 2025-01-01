import jittor as jt
import jittor.nn as nn
import numpy as np
import math


import sys
import io, os
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from model import BertForCL, BertConfig
# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import matplotlib.pyplot as plt

def infoNCE(sim, batch_size, temperature, alpha_neg=1.0):
    pos_indices = jt.concat([jt.arange(batch_size) + batch_size, jt.arange(batch_size)]).detach()

    # logits = sim / temperature
    logits = sim / temperature

    losses = []

    for i in range(batch_size):
        logits_i = jt.exp(logits[i])
        # print(logits_i)
        pos_col = pos_indices[i]

        sum_i = logits_i[batch_size:].sum()

        if logits.shape[0] == 3 * batch_size:
            # 有监督，调节hard negative的权重
            sum_i += logits_i[2 * batch_size:].sum() * (alpha_neg - 1.0)

        numerator_i = logits_i[pos_col]
        loss_i = jt.log(sum_i) -  jt.log(numerator_i)
        losses.append(loss_i)

    L = jt.stack(losses).mean()
    return L

def dynamic_infoNCE(sim, batch_size, temperature):
    ones_matrix = jt.ones_like(sim[0: batch_size, 0: batch_size])
    # 惩罚矩阵不参与梯度计算
    penalty_matrix = jt.exp(ones_matrix - sim[0: batch_size, 0: batch_size]).detach()

    logits = sim / temperature
    logits = logits[0: batch_size, batch_size:] * penalty_matrix

    losses = []

    for i in range(batch_size):
        logits_i = jt.exp(logits[i])
        pos_col = i
        
        sum_i = logits_i.sum()
        numerator_i = logits_i[pos_col]
        loss_i = jt.log(sum_i) -  jt.log(numerator_i)
        losses.append(loss_i)
    
    L = jt.stack(losses).mean()
    return L

        

def calc_loss(training_args, z1, z2, z3=None):
    if training_args.mode == "unsupervised":
        z = jt.contrib.concat([z1, z2], dim=0)
        sim = jt.matmul(z, z.transpose(0, 1))
        norm = z.norm(dim=1, keepdims=True)
        sim = sim / (norm * norm.transpose(0,1) + 1e-12)
        # sim = sim / training_args.temperature

        batch_size = z1.shape[0]

        # loss = infoNCE(sim, batch_size, training_args.temperature, training_args.negative_penalty)
        loss = dynamic_infoNCE(sim, batch_size, training_args.temperature)

        return loss
    elif training_args.mode == "supervised":
        z = jt.contrib.concat([z1, z2, z3], dim=0)

        sim = jt.matmul(z, z.transpose(0, 1))
        norm = z.norm(dim=1, keepdims=True)
        sim = sim / (norm * norm.transpose(0,1) + 1e-12)

        batch_size = z1.shape[0]

        loss = infoNCE(sim, batch_size, training_args.temperature, training_args.negative_penalty)
        return loss
    else:
        raise ValueError("The mode must be \"supervised\" or \"unsupervised\".")


def cosine_similarity(x, y):
    # 计算点积
    dot_product = (x * y).sum(dim=1)  # 对每个样本的维度求和，得到形状 (B,)
    
    # 计算 L2 范数
    x_norm = jt.sqrt((x ** 2).sum(dim=1))  # 形状 (B,)
    y_norm = jt.sqrt((y ** 2).sum(dim=1))  # 形状 (B,)
    
    # 计算余弦相似度
    cosine_sim = dot_product / (x_norm * y_norm)  # 形状 (B,)
    
    return cosine_sim




def evaluate(model, tokenizer_dir):
    # Set up the tasks
    task = 'STSBenchmark'
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                        'tenacity': 5, 'epoch_size': 4}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        max_length = 32
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='np',
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        model.eval()
        outputs = model(**batch)
        (encoder_output, pooled_output) = outputs

        return torch.from_numpy(pooled_output.cpu().numpy())

    results = {}

    se = senteval.engine.SE(params, batcher, prepare)
    result = se.eval(task)
    model.train()

    return result['dev']['spearman'].correlation * 100

def plot(loss_list, correlation_list, save_path):
    steps = [i * 50 for i in range(len(loss_list))]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, loss_list, label="Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Steps")
    plt.legend()
    plt.grid(True)
    plt.xticks([])

    plt.subplot(1, 2, 2)
    plt.plot(steps, correlation_list, label="Correlation", color="red")
    plt.xlabel("Steps")
    plt.ylabel("Correlation")
    plt.title("Correlation over Steps")
    plt.legend()
    plt.grid(True)
    plt.xticks([])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()