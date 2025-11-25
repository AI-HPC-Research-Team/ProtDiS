import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import random
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data

from .modules import FCHead, Swish, MLP
from .cdconv import CDConvEncoder, CDConvUNet
from .gnn import GNN
from .transformer import TransformerBlock
from .aggregator import Aggregator, FeatureFusion
from .input_layer import FeatureEmbedding


def build_encoder(cfg: DictConfig):
    if cfg.type == 'transformer':
       return TransformerBlock(
            cfg.n_heads,
            cfg.v_heads,
            cfg.n_layers,
            cfg.d_model,
        )
    elif cfg.type == 'gnn':
        return GNN(
            cfg.embed_dim,
            cfg.d_model,
            cfg.n_layers,
            cfg.dropout,
            cfg.gnn_type,
            cfg.use_edge_attr,
            cfg.pe,
        )
    elif cfg.type == 'cdconv':
        return CDConvEncoder(
            cfg.geometric_radii,
            cfg.sequential_kernel_size,
            cfg.kernel_channels,
            cfg.channels,
            cfg.base_width,
            cfg.embed_dim,
            cfg.d_model,
            cfg.get('pooling', 'mean'),
        )
    elif cfg.type == 'cdconv_unet':
        return CDConvUNet(
            cfg.geometric_radii,
            cfg.sequential_kernel_size,
            cfg.kernel_channels,
            cfg.channels,
            cfg.base_width,
            cfg.d_model,
        )
    elif cfg.type == 'mlp':
        return nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.d_model),
            *[MLP(cfg.d_model) for _ in range(cfg.n_layers)]
        )
    else:
        raise ValueError(f"Unknown encoder type: {cfg.type}")

def get_loss_fn(task_type: str) -> Callable:
    if task_type == 'classification':
        return nn.CrossEntropyLoss()
    elif task_type in ['multilabel', 'binary']:
        return nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        return nn.L1Loss()

class TaskModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_loss(self, data:Data, return_out=False):
        device = next(self.parameters()).device
        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        out = self.forward(data, all_loss)

        if self.task_type == 'classification':
            loss = self.loss_fn(out, data.y.long())
        elif self.task_type == 'multilabel':
            loss = self.loss_fn(out, data.y.float())
        elif self.task_type == 'binary':
            mask = data.y != -1  # mask out -1 labels
            loss = self.loss_fn(out.squeeze(-1)[mask], data.y[mask].float())
        elif self.task_type == 'regression':
            loss = self.loss_fn(out.squeeze(-1), data.y.float())
        else:
            raise ValueError(f"Unsupported task: {self.task_type}")

        all_loss += loss
        if return_out:
            return all_loss, out
        else:
            return all_loss
        
    def get_feature_importance(self):
        """
        Returns the attention weights from the last forward pass.
        Shape: (batch_size, num_features)
        """
        if self.feature_importance is not None:
            return {fea: imp for fea, imp in zip(self.input_encs, self.feature_importance.detach().cpu().numpy().mean(axis=0))}
        return None
    

class SingleModel(TaskModel):
    def __init__(self,
                 encoder: DictConfig,
                 embed_dim: int = 16,
                 d_model: int = 256,
                 feas: list = ['seq'],
                 input_dims: list = None,
                 drop_feats = None,
                 feature_fusion: str = 'gate',
                 branch_drop: float = 0.5,
                 pooling: str = 'mean',
                 task_type: list = ['protein', 'classification'],
                 num_classes: int = 7,
                 other_feature_dim: int = None,
                 aggregation: str = 'dot',) -> nn.Module:
        super().__init__()
        assert task_type[0] in ['protein', 'protein_pair', 'residue', 'residue_pair'], f"Unknown task: {task_type[0]}"
        assert task_type[1] in ['classification', 'multilabel', 'binary', 'regression'], f"Unknown task: {task_type[1]}"
        self.pair = 'pair' in task_type[0]  # 是否为成对蛋白质任务
        self.task_level = task_type[0]  # 任务级别
        self.task_type = task_type[1]  # 任务类型

        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.input_encs = feas.copy()
        if 'seq' in self.input_encs:
            self.input_encs.remove('seq')  # 序列特征单独处理
            self.seq_embeddings = nn.Embedding(21, embed_dim)
        self.num_features = len(self.input_encs)
        self.branch_drop = nn.Dropout(branch_drop)

        assert hasattr(self, 'seq_embeddings') or self.num_features > 0, "At least one of sequence or other features must be provided."
        if self.num_features > 1:
            self.feature_fusion = FeatureFusion(fusion_method=feature_fusion, d_model=embed_dim, num_features=self.num_features)
        if self.num_features > 0:
            self.embeddings = FeatureEmbedding(embed_dim, self.input_encs, self.input_dims, drop_feats)
            if hasattr(self, 'seq_embeddings'):
                self.final_fuser = Aggregator(embed_dim, aggregation='concat')

        self.encoder = build_encoder(encoder)

        if 'protein' in self.task_level:
            assert pooling in ['mean', 'add', 'max'], f"Unknown pooling method: {pooling}"
            self.global_pool = eval(f'global_{pooling}_pool')

        if other_feature_dim is not None and other_feature_dim > 0:
            self.other_encoder = nn.Sequential(
                nn.Linear(other_feature_dim, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(True),
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
            )
            self.other_aggregator = Aggregator(d_model, aggregation)

        if self.pair: self.aggregator = Aggregator(d_model, aggregation)

        num_classes = num_classes if self.task_type in ['classification', 'multilabel'] else 1
        self.task_head = nn.Linear(d_model, num_classes)
        self.loss_fn = get_loss_fn(self.task_type)
        
        # 存储特征重要性用于可视化
        self.feature_importance = None

    def forward(self, data:Data, all_loss=None):
        if hasattr(self, 'seq_embeddings'):
            seq_embs = self.seq_embeddings(data.seq)  # (num_nodes, embed_dim)

        if self.num_features > 0:
            feature_embs = self.embeddings(data)  # list[(num_nodes, embed_dim)]
            if self.num_features > 1: # 使用融合模块进行特征融合
                fused_x, self.feature_importance = self.feature_fusion(feature_embs, s_embeddings=data.s)
            else: # 如果只有一个特征，直接使用该特征
                fused_x, self.feature_importance = feature_embs[0], None
        else:
            fused_x, self.feature_importance = seq_embs, None

        if hasattr(self, 'seq_embeddings') and self.num_features > 0:
            fused_x = self.final_fuser(seq_embs, self.branch_drop(fused_x))  # 最终融合后的特征            

        # 将融合后的特征输入编码器
        if self.pair:  # 考虑存在成对蛋白质的情况
            data.batch = 2 * data.batch + data.p_idx
        out = self.encoder(fused_x, data)  # (num_nodes, d_model)

        if 'protein' in self.task_level:
            # 图级表示
            if out.shape[0] == data.batch.max() + 1:
                graph_repr = out
            else:
                graph_repr = self.global_pool(out, data.batch)

            if self.pair:
                # 如果是成对蛋白质，取平均
                graph_repr = graph_repr.view(-1, 2, graph_repr.size(-1))
                graph_repr = self.aggregator(graph_repr[:, 0], graph_repr[:, 1])
            
            if hasattr(self, 'other_encoder'):
                other_features = self.other_encoder(data.other)
                graph_repr = self.other_aggregator(graph_repr, other_features)

            # 图级输出
            out = self.task_head(graph_repr)
        else:
            if self.pair:
                pairs = []
                for i in range(0, data.batch.max(), 2):
                    x0 = out[data.batch == i]        # [n_i, d_model]
                    x1 = out[data.batch == i + 1]  # [n_(i+1), d_model]

                    x0_exp = x0.unsqueeze(1)           # [n_i, 1, d_model]
                    x1_exp = x1.unsqueeze(0)         # [1, n_j, d_model]

                    combined = torch.cat([x0_exp.expand(-1, x1_exp.size(1), -1),
                                        x1_exp.expand(x0_exp.size(0), -1, -1)], dim=2)
                    combined = combined.view(-1, 2 * self.d_model)
                    pairs.append(combined)

                pairs = torch.cat(pairs, dim=0)  # [total_pairs, 2 * d_model]
                mask = data.y != -1  # mask out -1 labels
                pairs = pairs[mask]  # 只保留有效的成对数据
                data.y = data.y[mask]  # 只保留有效的标签
                out = self.aggregator(pairs[:, :self.d_model], pairs[:, self.d_model:])  # [total_pairs, d_model]

            # 节点级输出
            out = self.task_head(out)

        return out

class ProbeModel(TaskModel):
    def __init__(self,
                 d_model: int = 128,
                 layers: int = 5,
                 input_encs: list = None,
                 input_dims: list = None,
                 task_type: list = ['protein', 'classification'],
                 pooling: str = 'mean',
                 num_classes: int = 7,
                 other_feature_dim: int = None,
                 aggregation: str = 'dot',) -> nn.Module:
        super().__init__()
        assert task_type[0] in ['protein', 'protein_pair', 'residue', 'residue_pair'], f"Unknown task: {task_type[0]}"
        assert task_type[1] in ['classification', 'multilabel', 'binary', 'regression'], f"Unknown task: {task_type[1]}"
        self.pair = 'pair' in task_type[0]  # 是否为成对蛋白质任务
        self.task_level = task_type[0]  # 任务级别
        self.task_type = task_type[1]  # 任务类型

        self.input_encs = input_encs or ['seq', 'ck', 'ss', 'asa', 'bfactor', 'lp', 'hydro', 'stability', 'entropy', 'orc']
        if input_dims is None:
            input_dims = [128] * len(self.input_encs)
        self.d_model = d_model
        self.num_features = len(self.input_encs)

        # 每个特征先映射到同一维度 d_model
        self.feature_maps = nn.ModuleList()
        for fea, dim in zip(self.input_encs, input_dims):
            if fea == 'seq':
                self.feature_maps.append(nn.Embedding(21, d_model))
            else:
                self.feature_maps.append(
                    nn.Sequential(
                        nn.Linear(dim, d_model),
                        nn.LayerNorm(d_model),
                        nn.ReLU())
                )

        layer_list = [
            nn.Linear(len(input_dims) * d_model, d_model),
            *[MLP(d_model) for _ in range(layers)]
        ]
        self.encoder = nn.Sequential(*layer_list)

        if 'protein' in self.task_level:
            assert pooling in ['mean', 'add', 'max'], f"Unknown pooling method: {pooling}"
            self.global_pool = eval(f'global_{pooling}_pool')

        if other_feature_dim is not None and other_feature_dim > 0:
            self.other_encoder = nn.Sequential(
                nn.Linear(other_feature_dim, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(True),
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
            )
            self.other_aggregator = Aggregator(d_model, aggregation)

        if self.pair: self.aggregator = Aggregator(d_model, aggregation)

        num_classes = num_classes if self.task_type in ['classification', 'multilabel'] else 1
        self.task_head = nn.Linear(d_model, num_classes)
        self.loss_fn = get_loss_fn(self.task_type)

        self.feature_importance = None

    def forward(self, data:Data, all_loss=None):
        # 先对每个特征单独映射和归一化
        processed = []
        for enc, fmap in zip(self.input_encs, self.feature_maps):
            x = getattr(data, enc)
            x = fmap(x)
            processed.append(x)
        fused_x = torch.cat(processed, dim=-1)

        # 将融合后的特征输入编码器
        if self.pair:  # 考虑存在成对蛋白质的情况
            data.batch = 2 * data.batch + data.p_idx
        out = self.encoder(fused_x)  # (num_nodes, d_model)

        if 'protein' in self.task_level:
            # 图级表示
            if out.shape[0] == data.batch.max() + 1:
                graph_repr = out
            else:
                graph_repr = self.global_pool(out, data.batch)

            if self.pair:
                # 如果是成对蛋白质，取平均
                graph_repr = graph_repr.view(-1, 2, graph_repr.size(-1))
                graph_repr = self.aggregator(graph_repr[:, 0], graph_repr[:, 1])
            
            if hasattr(self, 'other_encoder'):
                other_features = self.other_encoder(data.other)
                graph_repr = self.other_aggregator(graph_repr, other_features)

            # 图级输出
            out = self.task_head(graph_repr)
        else:
            if self.pair:
                pairs = []
                for i in range(0, data.batch.max(), 2):
                    x0 = out[data.batch == i]        # [n_i, d_model]
                    x1 = out[data.batch == i + 1]  # [n_(i+1), d_model]

                    x0_exp = x0.unsqueeze(1)           # [n_i, 1, d_model]
                    x1_exp = x1.unsqueeze(0)         # [1, n_j, d_model]

                    combined = torch.cat([x0_exp.expand(-1, x1_exp.size(1), -1),
                                        x1_exp.expand(x0_exp.size(0), -1, -1)], dim=2)
                    combined = combined.view(-1, 2 * self.d_model)
                    pairs.append(combined)

                pairs = torch.cat(pairs, dim=0)  # [total_pairs, 2 * d_model]
                mask = data.y != -1  # mask out -1 labels
                pairs = pairs[mask]  # 只保留有效的成对数据
                data.y = data.y[mask]  # 只保留有效的标签
                out = self.aggregator(pairs[:, :self.d_model], pairs[:, self.d_model:])  # [total_pairs, d_model]

            # 节点级输出
            out = self.task_head(out)

        return out
