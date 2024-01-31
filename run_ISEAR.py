import time
import torch
import numpy as np
from train_eval_isear import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import os
import random

def split_ISEAR_total_file(file_path, output_dir, split_index):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 将文件分成5部分
    chunk_size = len(lines) // 5
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # 选择验证集y和训练集c
    y = chunks[split_index]
    c = [item for index, chunk in enumerate(chunks) if index != split_index for item in chunk]

    # 从c中随机选择20%作为训练集用的测试集cy
    random.shuffle(c)
    cy_size = len(c) // 5
    cy = c[:cy_size]
    c = c[cy_size:]

    # 写入到相应的文件
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as file:
        file.writelines(y)
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as file:
        file.writelines(c)
    with open(os.path.join(output_dir, 'dev.txt'), 'w', encoding='utf-8') as file:
        file.writelines(cy)

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':

    for i in range(5):
        dataset = 'ISEAR'  # 数据集
        model_name = args.model
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        np.random.seed(3407)
        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        start_time = time.time()
        print("Loading data...")
        split_ISEAR_total_file(os.path.join(dataset+config.current_dataset, 'total.txt'),dataset+config.current_dataset,i)
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter, test_iter)

