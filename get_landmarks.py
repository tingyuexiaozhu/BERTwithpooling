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
from pytorch_pretrained import BertModel, BertTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()
    dataset = 'goemotion'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    train_data, dev_data, test_data = build_dataset(config)

    dataI =DatasetIterater(train_data, 50, config.device)
    x, y=dataI._to_tensor(train_data)
    context=x[0]
    mask=x[2]
    bert=BertModel.from_pretrained(config.bert_path).to(config.device)
    n=context.shape[0]

    batch_size = 10

    # 存储所有批次的 embeddings
    all_embeddings = []

    # 循环处理每个批次
    for i in tqdm(range(0, context.size(0), batch_size), desc="Processing batches"):
        # 对最后一个批次进行处理，确保包含所有剩余的项
        batch_context = context[i:i + batch_size] if i + batch_size < context.size(0) else context[i:]
        batch_mask = mask[i:i + batch_size] if i + batch_size < mask.size(0) else mask[i:]

        # 获取每个批次的 embeddings，并直接合并
        with torch.no_grad():  # 确保不保存梯度信息，减少显存使用
            batch_embeddings = bert(batch_context, attention_mask=batch_mask,
                                    output_all_encoded_layers=False)[0]
            all_embeddings.append(batch_embeddings.cpu())  # 将 embeddings 移至 CPU

        # 清除不再需要的变量以释放显存
        del batch_context, batch_mask, batch_embeddings
        torch.cuda.empty_cache()  # 清理未使用的缓存

    # 合并所有批次的 embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)

    nums, seq_len, hidden_size = final_embeddings.shape
    encoder_out = final_embeddings.view(nums * seq_len, hidden_size).cpu().detach().numpy()

    kmeans = KMeans(n_clusters=200, random_state=0, n_init=10, max_iter=300).fit(encoder_out)
    cluster_centers = kmeans.cluster_centers_
    np.save('new_cluster_centers200.npy', cluster_centers)


