import jittor as jt
import jittor.nn as nn
import numpy as np
import math


# 采用CrossEntropyLoss计算余弦相似度
def calc_loss(training_args, z1, z2, z3=None):
    if training_args.mode == "unsupervised":
        # 无监督条件下与论文公式不完全一样，而是将batch内的
        # 其余例子也作为了负例
        z = jt.contrib.concat([z1, z2], dim=0)
        sim = jt.matmul(z, z.transpose(0, 1))
        norm = z.norm(dim=1, keepdims=True)
        sim = sim / (norm * norm.transpose(0,1) + 1e-12)
        sim = sim / training_args.temperature

        batch_size = z1.shape[0]
        labels = np.array([i + batch_size if i < batch_size else i - batch_size for i in range(2*batch_size)])
        labels = jt.array(labels, dtype="int64")

        loss_tensor = nn.cross_entropy_loss(sim, labels, reduction="none")
        weight = jt.ones([2 * batch_size])
        weighted_loss = loss_tensor * weight
        loss = weighted_loss.mean()
        return loss
    elif training_args.mode == "supervised":
        z = jt.contrib.concat([z1, z2, z3], dim=0)

        sim = jt.matmul(z, z.transpose(0, 1))
        norm = z.norm(dim=1, keepdims=True)
        sim = sim / (norm * norm.transpose(0,1) + 1e-12)
        sim = sim / training_args.temperature

        batch_size = z1.shape[0]
        labels = np.array([i + batch_size if i < batch_size else i - batch_size for i in range(3*batch_size)])
        labels = jt.array(labels, dtype="int64")

        # ce_loss_fn = nn.CrossEntropyLoss()
        # loss_tensor = ce_loss_fn(sim, labels)
        loss_tensor = nn.cross_entropy_loss(sim, labels, reduction="none")
        
        weight = jt.ones([3 * batch_size])
        weight[2 * batch_size:] = 0.0
        weight[:2 * batch_size] *= 1.5

        weighted_loss = loss_tensor * weight
        loss = weighted_loss.mean()

        return loss
    else:
        raise ValueError("The mode must be \"supervised\" or \"unsupervised\".")
