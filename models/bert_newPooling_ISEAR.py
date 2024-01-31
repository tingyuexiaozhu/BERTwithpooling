import torch
import torch.nn as nn
import os
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = '1e-5 batch size8 pad60 ISEAR_exclude pad bert base uncased   without dropout before gcn_200_cos_0.3learnable_nonlinear in graph constructing_76851212832'
        self.current_dataset = '/ISEAR'
        self.train_path = dataset + self.current_dataset + '/train.txt'  # 训练集
        self.dev_path = dataset + self.current_dataset + '/dev.txt'  # 验证集
        self.test_path = dataset + self.current_dataset + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + self.current_dataset + '/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数=
        self.num_epochs = 1000  # epoch数
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 60  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.bert_path = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size1 = 768
        self.hidden_size2 = 512
        self.hidden_size3 = 128
        self.hidden_size4 = 32

        self.dropout = 0.5
        self.num_clusters = 200
        self.unfreeze_epoch = 3  # 何时更新BERT权重
        # self.checkpoint_dir = '../geoemotion/saved_dict'
        self.resume_from_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   'ISEAR', 'saved_dict')


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.d = nn.Parameter(torch.tensor([0.3])).to(config.device)  # 高斯核函数带宽
        self.dropout = nn.Dropout(config.dropout)
        self.num_clusters = config.num_clusters
        self.fc1 = nn.Linear(config.hidden_size1 * self.num_clusters, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, config.num_classes)
        self.landmarks_initialized = False
        self.landmarks = None
        self.device = config.device
        self.batch_size = config.batch_size
        self.gcn1 = GCNConv(config.hidden_size1, config.hidden_size2)
        self.gcn2 = GCNConv(config.hidden_size2, config.hidden_size3)
        self.gcn3 = GCNConv(config.hidden_size3, config.hidden_size4)
        self.padsize = config.pad_size

        # self.position_embedding_layer = nn.Embedding(config.pad_size, config.num_clusters)
        # self.position_embedding_layer = nn.Embedding(config.pad_size, 1)
        self.clusters_ = "ISEAR_new_cluster_centers200_without pad.npy"
        self.alpha_for_distanceweight = nn.Parameter(torch.tensor([0.5])).to(config.device)
        self.hidden_size4 = 32
        self.num_clusters = config.num_clusters
        self.adjust_dim = nn.Linear(config.hidden_size4, config.hidden_size1)

        self.beta= nn.Parameter(torch.tensor([1.0])).to(config.device) #计算图权重时非线性的参数

    def initialize_landmarks(self):
        # 使用 K-Means 初始化 landmarks
        # 假设这里的 data 是您模型的输入数据
        print("-------------------------------------------------------initialize_landmarks!!!!!!!!!!!!!!!!!!!!!!!!!!")
        initial_landmarks = np.load(self.clusters_)

        # 将 numpy 数组转换为 PyTorch tensor，并设置为可训练参数
        self.landmarks = nn.Parameter(torch.tensor(initial_landmarks, dtype=torch.float32))
        self.landmarks_initialized = True

    def distance_matrix(self, data, clusters):

        data_expanded = data.unsqueeze(2)  # shape: (b, n, 1, k)
        clusters_expanded = clusters.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, m, k)
        data_expanded = data_expanded.to(self.device)
        clusters_expanded = clusters_expanded.to(self.device)
        # 计算差的平方
        diff_square = (data_expanded - clusters_expanded) ** 2

        # 求和并开方以得到欧氏距离
        distance = torch.sqrt(torch.sum(diff_square, dim=-1))

        min_vals = distance.min(dim=-1, keepdim=True)[0]
        max_vals = distance.max(dim=-1, keepdim=True)[0]

        # 避免除以零的情况，可以添加一个小的epsilon
        epsilon = 1e-6
        normalized_distance = (distance - min_vals) / (max_vals - min_vals + epsilon)

        return normalized_distance

    def probability_matrix(self, data, pad_mask):

        similarity_matrix = torch.exp(- (data ** 2) / (2 * self.d ** 2))

        # 将pad_mask扩展到b*n*d
        expanded_pad_mask = pad_mask.unsqueeze(-1).expand_as(similarity_matrix)

        # 将similarity_matrix中对应pad_mask为0的位置置为0
        similarity_matrix = similarity_matrix * expanded_pad_mask

        # 将相似度矩阵转换为概率矩阵
        # 沿着每个点的聚类中心维度进行归一化
        probability_matrix = similarity_matrix / (similarity_matrix.sum(dim=1, keepdim=True)+1e-6)

        return probability_matrix

    def graph_constructing(self, data, distance_weight):
        b, k, n = data.shape

        # 计算外积
        outer_product = torch.einsum('bik,bjl->bijkl', data, data)

        # 重复distance_weight以匹配外积张量的大小
        distance_weight_expanded = distance_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, k, k, 1, 1)

        # 进行Hadamard乘积
        Q = outer_product * distance_weight_expanded

        Q = torch.exp(self.beta * Q) - 1
        # 对Q的每个n x n矩阵求和，获得b x k x k x 1的张量
        G = Q.sum(dim=-1).sum(dim=-1)

        return G

    def compute_distance_weight(self):
        n = self.padsize
        indices = torch.arange(n).unsqueeze(0)
        diff = torch.abs(indices - indices.T)
        smooth_matrix = torch.exp(-self.alpha_for_distanceweight * diff.float().to(self.device))
        return smooth_matrix

    def gcn_batched(self, data, weights):
        data_ori = data
        B, K, D = data.shape
        graphs = []
        for i in range(B):
            x = data[i]
            edge_index = weights[i].nonzero(as_tuple=False).t().contiguous()
            graphs.append(Data(x=x, edge_index=edge_index))

        batched_data = Batch.from_data_list(graphs)

        # 使用三个GCN层
        out = self.gcn1(batched_data.x, batched_data.edge_index)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.gcn2(out, batched_data.edge_index)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.gcn3(out, batched_data.edge_index)
        out = F.relu(out)
        out = self.dropout(out)

        out = out.view(-1, self.hidden_size4)  # 形状变为 [B*K, hidden_size4]
        out = self.adjust_dim(out)  # 维度调整为 [B*K, M]
        out = out.view(B, K, -1)

        return out + data_ori

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out = self.bert(context, attention_mask=mask).last_hidden_state
        if not self.landmarks_initialized:
            self.initialize_landmarks()

        distance_matrix = self.distance_matrix(encoder_out, self.landmarks)
        probability_matrix = self.probability_matrix(distance_matrix,mask)
        distance_weight = self.compute_distance_weight()

        weight_graph = self.graph_constructing(probability_matrix.permute(0, 2, 1), distance_weight)
        out = torch.bmm(probability_matrix.transpose(1, 2), encoder_out)
        # out_ori = out
        # out = self.dropout(out)
        out = self.gcn_batched(out, weight_graph)
        # out = out_ori + out
        out = out.reshape(out.size(0), -1)
        x = out
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x