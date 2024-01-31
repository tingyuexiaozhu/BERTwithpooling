import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from sklearn.cluster import KMeans

from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import os
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import argparse
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif
from utils import DatasetIterater
from transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()
    dataset = 'ISEAR'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    train_data, _, _ = build_dataset(config)

    dataI = DatasetIterater(train_data, 50, config.device)
    x, y = dataI._to_tensor(train_data)
    context = x[0]
    mask = x[2]
    bert = BertModel.from_pretrained('bert-base-uncased').to(config.device)
    n = context.shape[0]

    batch_size = 20

    # 存储所有批次的 embeddings
    all_embeddings = []
    non_pad_indices = []

    # 循环处理每个批次
    for i in tqdm(range(0, context.size(0), batch_size), desc="Processing batches"):
        # 处理当前批次
        end_idx = min(i + batch_size, context.size(0))
        batch_context = context[i:end_idx]
        batch_mask = mask[i:end_idx]

        # 记录非pad部分的索引（二维索引）
        non_pad_indices_batch = (batch_mask != 0).nonzero(as_tuple=False)
        non_pad_indices_batch = non_pad_indices_batch.to(config.device)
        # 调整索引，考虑批次的起始位置
        non_pad_indices_batch += torch.tensor([i, 0]).to(config.device)

        non_pad_indices.extend(non_pad_indices_batch.tolist())

        # 获取每个批次的embeddings
        with torch.no_grad():
            batch_embeddings = bert(batch_context, attention_mask=batch_mask).last_hidden_state
            all_embeddings.append(batch_embeddings.cpu())

        # 清除不再需要的变量以释放显存
        del batch_context, batch_mask, batch_embeddings
        torch.cuda.empty_cache()

    # 合并所有批次的embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)

    # 计算扁平化后非填充词的索引
    seq_len = final_embeddings.size(1)
    flat_indices = [idx[0] * seq_len + idx[1] for idx in non_pad_indices]

    # 使用一维索引获取non_pad_embeddings
    hidden_size = final_embeddings.size(2)
    non_pad_embeddings = final_embeddings.view(-1, hidden_size)[flat_indices]

    kmeans = KMeans(n_clusters=200, random_state=0, n_init=10, max_iter=300).fit(non_pad_embeddings)
    cluster_centers = kmeans.cluster_centers_
    np.save('ISEAR_new_cluster_centers200_without pad.npy', cluster_centers)
