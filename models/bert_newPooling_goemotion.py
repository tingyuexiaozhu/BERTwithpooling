import torch
import torch.nn as nn

from pytorch_pretrained import BertModel, BertTokenizer
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'morelandmarks'
        self.current_dataset = '/original'
        self.train_path = dataset + self.current_dataset + '/train.txt'  # 训练集
        self.dev_path = dataset + self.current_dataset + '/dev.txt'  # 验证集
        self.test_path = dataset + self.current_dataset + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + self.current_dataset + '/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数=
        self.num_epochs = 100  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-6  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.5
        self.num_clusters = 100
        self.unfreeze_epoch = 3  # 何时更新BERT权重
        self.checkpoint_dir = './models/morelandmarks'
        self.resume_from_checkpoint = './models/morelandmarks'


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, q, k, v):
        q = self.query(q).unsqueeze(0)  # Add batch dimension
        k = self.key(k).permute(1, 0, 2)  # Convert to (seq_len, batch, dim)
        v = self.value(v).permute(1, 0, 2)
        output, _ = self.attention(q, k, v, need_weights=False)
        return output.squeeze(0)  # Remove batch dimension


class CrossAttention_ComputeLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention_ComputeLayer, self).__init__()
        self.cross_attention = CrossAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(x.size(1)):
            q = x[:, i, :]
            k = torch.cat([x[:, :i, :], x[:, i + 1:, :]], dim=1)
            v = k
            attention_output = self.cross_attention(q, k, v)
            out[:, i, :] = attention_output + x[:, i, :]  # Skip Connection
        return self.norm(out)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.d = 0.3  # 高斯核函数带宽
        self.dropout = nn.Dropout(config.dropout)
        self.num_clusters = config.num_clusters
        self.fc1 = nn.Linear(config.hidden_size * self.num_clusters, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, config.num_classes)
        self.landmarks_initialized = False
        self.landmarks = None
        self.device = config.device
        self.batch_size = config.batch_size
        self.crossattention = CrossAttention_ComputeLayer(config.hidden_size)
        self.gcn1 = GCNConv(config.hidden_size, config.hidden_size)
        self.gcn2 = GCNConv(config.hidden_size, config.hidden_size)
        self.gcn3 = GCNConv(config.hidden_size, config.hidden_size)
        self.padsize = config.pad_size

        # self.position_embedding_layer = nn.Embedding(config.pad_size, config.num_clusters)
        # self.position_embedding_layer = nn.Embedding(config.pad_size, 1)
        self.clusters_ = "cluster_centers.npy"
        self.alpha_for_distanceweight = nn.Parameter(torch.tensor([0.5])).to(config.device)

    def initialize_landmarks(self):
        # 使用 K-Means 初始化 landmarks
        # 假设这里的 data 是您模型的输入数据

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

    def probability_matrix(self, data):
        similarity_matrix = torch.exp(- (data ** 2) / (2 * self.d ** 2))

        # 将相似度矩阵转换为概率矩阵
        # 沿着每个点的聚类中心维度进行归一化
        probability_matrix = similarity_matrix / similarity_matrix.sum(dim=-1, keepdim=True)
        return probability_matrix

    def graph_constructing(self, data, distance_weight):
        b, k, n = data.shape

        # 计算外积
        outer_product = torch.einsum('bik,bjl->bijkl', data, data)

        # 重复distance_weight以匹配外积张量的大小
        distance_weight_expanded = distance_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, k, k, 1, 1)

        # 进行Hadamard乘积
        Q = outer_product * distance_weight_expanded

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
        out = self.gcn2(out, batched_data.edge_index)
        out = self.gcn3(out, batched_data.edge_index)

        out = out.view(B, K, D)

        return out + data_ori

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        if not self.landmarks_initialized:
            print(
                "-------------------------------------------------------self.landmarks_initialized!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.initialize_landmarks(encoder_out)

        distance_matrix = self.distance_matrix(encoder_out, self.landmarks)
        probability_matrix = self.probability_matrix(distance_matrix)
        distance_weight = self.compute_distance_weight()

        weight_graph = self.graph_constructing(probability_matrix.permute(0, 2, 1), distance_weight)
        out = torch.bmm(probability_matrix.transpose(1, 2), encoder_out)
        out_ori = out
        out = self.dropout(out)
        out = self.gcn_batched(out, weight_graph)
        out = out_ori + out
        out = out.reshape(out.size(0), -1)
        x = out
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
