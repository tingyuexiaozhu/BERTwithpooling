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

    dataI =DatasetIterater(train_data, config.batch_size, config.device)
    x, y=dataI._to_tensor(train_data)
    context=x[0]
    mask=x[2]
    bert=BertModel.from_pretrained(config.bert_path).to(config.device)
    n=context.shape[0]
    num_rows_to_select = n // 50

    # 随机生成索引
    indices = torch.randperm(n)[:num_rows_to_select]

    # 使用这些索引来抽取context和mask的行
    selected_context = context[indices]
    selected_mask = mask[indices]

    encoder_out, text_cls = bert(selected_context, attention_mask=selected_mask, output_all_encoded_layers=False)
    nums, seq_len, hidden_size = encoder_out.shape
    encoder_out = encoder_out.view(nums*seq_len,  hidden_size).cpu().detach().numpy()

    kmeans = KMeans(n_clusters=100, random_state=0, n_init=10, max_iter=300).fit(encoder_out)
    cluster_centers = kmeans.cluster_centers_
    np.save('cluster_centers.npy', cluster_centers)
