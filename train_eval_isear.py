# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import os
from torch.optim import AdamW  # 确保已经导入AdamW



# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# 原来的
# def train(config, model, train_iter, dev_iter, test_iter):
#     start_time = time.time()
#     model.train()
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
#     # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
#     optimizer = BertAdam(optimizer_grouped_parameters,
#                          lr=config.learning_rate,
#                          warmup=0.05,
#                          t_total=len(train_iter) * config.num_epochs)
#     total_batch = 0  # 记录进行到多少batch
#     dev_best_loss = float('inf')
#     last_improve = 0  # 记录上次验证集loss下降的batch数
#     flag = False  # 记录是否很久没有效果提升
#     model.train()
#     for epoch in range(config.num_epochs):
#         print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         for i, (trains, labels) in enumerate(train_iter):
#             outputs = model(trains)
#             model.zero_grad()
#             loss = F.cross_entropy(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             if total_batch % 100 == 0:
#                 # 每多少轮输出在训练集和验证集上的效果
#                 true = labels.data.cpu()
#                 predic = torch.max(outputs.data, 1)[1].cpu()
#                 train_acc = metrics.accuracy_score(true, predic)
#                 dev_acc, dev_loss = evaluate(config, model, dev_iter)
#                 if dev_loss < dev_best_loss:
#                     dev_best_loss = dev_loss
#                     torch.save(model.state_dict(), config.save_path)
#                     improve = '*'
#                     last_improve = total_batch
#                 else:
#                     improve = ''
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
#                 print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
#                 model.train()
#             total_batch += 1
#             if total_batch - last_improve > config.require_improvement:
#                 # 验证集loss超过1000batch没下降，结束训练
#                 print("No optimization for a long time, auto-stopping...")
#                 flag = True
#                 break
#         if flag:
#             break
#     test(config, model, test_iter)

# def find_latest_checkpoint(checkpoint_dir):
#     """在指定的文件夹中找到最新的checkpoint文件"""
#     checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pt')]
#     if not checkpoint_files:
#         return None
#     latest_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
#     return os.path.join(checkpoint_dir, latest_file)
def find_file_with_name(directory_path, file_name):
    file_name+=".ckpt"
    # 检查文件夹是否存在
    if not os.path.exists(directory_path):
        print("------------folder not exist------------")
        return None

    # 获取文件夹中的所有文件
    files = os.listdir(directory_path)

    # 查找是否有与file_name相同名字的文件
    for f in files:
        if f == file_name:
            return os.path.join(directory_path, f)

    # 如果没有找到匹配的文件，则返回None
    return None

def split_data(data, fold_index, k=5):
    """将数据划分为训练集和验证集。
    data: 整个数据集
    fold_index: 当前的折数索引
    k: 总共的折数
    """
    fold_size = len(data) // k
    validation_data = data[fold_index * fold_size:(fold_index + 1) * fold_size]
    train_data = data[:fold_index * fold_size] + data[(fold_index + 1) * fold_size:]
    return train_data, validation_data


# mine 冻结BERT权重的
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()

    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 示例参数

    latest_checkpoint = find_file_with_name(config.resume_from_checkpoint, config.model_name)

    if latest_checkpoint:

        print(f"Loading checkpoint: {config.resume_from_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        if 'landmarks' in checkpoint and checkpoint['landmarks'] is not None:
            model.landmarks = nn.Parameter(checkpoint['landmarks'])
            model.landmarks_initialized = True
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 加载调度器状态

        start_epoch = checkpoint['epoch'] + 1
        total_batch = checkpoint['total_batch']  # 恢复batch计数
        dev_best_acc = checkpoint['dev_best_acc']
        last_improve = checkpoint['last_improve']

    else:

        # 冻结BERT层的权重
        for name, param in model.named_parameters():
            if 'bert' in name:
                param.requires_grad = False
        start_epoch = 0
        total_batch = 0
        dev_best_acc = 0
        last_improve = 0
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate,
                          weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 示例参数

    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(start_epoch, config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        if epoch == config.unfreeze_epoch:  # 判断是否到了解冻BERT层的epoch
            for name, param in model.named_parameters():
                if 'bert' in name:
                    param.requires_grad = True

            # 重新配置优化器以包含解冻的BERT层参数
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 示例参数

        model.train()

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, macro_f1, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    improve = '*'
                    last_improve = total_batch
                    checkpoint = {
                        'epoch': epoch,
                        'total_batch': total_batch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  # 保存调度器状态
                        'loss': loss,
                        'dev_best_acc': dev_best_acc,
                        'last_improve': last_improve,
                        'landmarks_initialized': model.landmarks_initialized,
                        'landmarks': model.landmarks if model.landmarks is not None else None
                    }

                    # 保存checkpoint
                    torch.save(checkpoint, config.save_path)

                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break


        if flag:
            break
    test(config, model, test_iter)


# def test(config, model, test_iter):
#     # test
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score...")
#     print(test_report)
#     print("Confusion Matrix...")
#     print(test_confusion)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)

def test(config, model, test_iter):
    # 加载模型
    # model.load_state_dict(torch.load(config.save_path))
    checkpoint = torch.load(config.save_path)

    # 提取模型状态字典
    model_state_dict = checkpoint['model_state_dict']

    # 加载模型状态
    model.load_state_dict(model_state_dict)
    model.eval()
    start_time = time.time()

    # 调用 evaluate 函数，现在它返回一个额外的 macro_f1 值
    test_acc, test_macro_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    # 打印测试损失和准确率
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Test Macro F1: {2:>6.2%}'
    print(msg.format(test_loss, test_acc, test_macro_f1))

    # 打印精确度、召回率和F1得分
    print("Precision, Recall and F1-Score...")
    print(test_report)

    # 打印混淆矩阵
    print("Confusion Matrix...")
    print(test_confusion)

    # 计算并打印测试所用的时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return test_acc, test_macro_f1, test_loss, test_report, test_confusion


# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     labels_all = np.array([], dtype=int)
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss
#             labels = labels.data.cpu().numpy()
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             labels_all = np.append(labels_all, labels)
#             predict_all = np.append(predict_all, predic)
#
#     acc = metrics.accuracy_score(labels_all, predict_all)
#     if test:
#         report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(labels_all, predict_all)
#         return acc, loss_total / len(data_iter), report, confusion
#     return acc, loss_total / len(data_iter)
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    macro_f1 = metrics.f1_score(labels_all, predict_all, average='macro')

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, macro_f1, loss_total / len(data_iter), report, confusion

    return acc, macro_f1, loss_total / len(data_iter)

