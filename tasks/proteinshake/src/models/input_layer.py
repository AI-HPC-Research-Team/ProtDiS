import torch
import torch.nn as nn
from typing import Optional
from torch_geometric.data import Data

from .modules import Swish, MLP

ASA_DISCRETIZATION_BOUNDARIES = [0, 0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4, 336.]
BFACTOR_DISCRETIZATION_BOUNDARIES = [-20, 8.6, 11.7, 14.1, 16.1, 18.0, 19.8, 21.6, 23.4, 25.3, 27.2, 29.2, 31.3, 33.5, 35.9, 38.4, 41.2, 44.2, 47.5, 51.4, 55.9, 61.4, 68.8, 80.6, 150., 570.]
LP_DISCRETIZATION_BOUNDARIES = [3.76, 6.0, 6.12, 6.21, 6.29, 6.36, 6.44, 6.51, 6.59, 6.68, 6.76, 6.85, 6.94, 7.03, 7.12, 7.21, 7.31, 7.42, 7.54, 7.69, 7.87, 8.1, 8.43, 9.05, 16.89]
HYDRO_DISCRETIZATION_BOUNDARIES = [-4.5, -2.12, -1.71, -1.44, -1.23, -1.05, -0.89, -0.74, -0.6, -0.47, -0.34, -0.22, -0.1, 0.02, 0.14, 0.27, 0.4, 0.54, 0.69, 0.85, 1.02, 1.24, 1.5, 1.88, 4.5]
STABILITY_DISCRETIZATION_BOUNDARIES = [0., 0.92, 1.08, 1.17, 1.23, 1.28, 1.32, 1.35, 1.38, 1.41, 1.44, 1.465, 1.49, 1.51, 1.53, 1.55, 1.575, 1.6, 1.62, 1.645, 1.67, 1.7, 1.74, 1.8, 5.13]
ENTROPY_DISCRETIZATION_BOUNDARIES = [0., 0.485, 0.544, 0.565, 0.590, 0.613, 0.637, 0.655, 0.667, 0.711, 0.721, 0.763, 0.931]

def rbf(values: torch.Tensor, boundaries: torch.Tensor):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    centers = (boundaries[:-1] + boundaries[1:]) / 2.0
    centers = centers.view([1] * len(values.shape) + [-1])
    rbf_std = (boundaries[1:] - boundaries[:-1]) / 2.0
    rbf_std = rbf_std.view([1] * len(values.shape) + [-1])
    z = (values.unsqueeze(-1) - centers) / rbf_std
    return torch.exp(-(z**2))

class FeatureEmbedding(nn.Module):
    def __init__(self, embed_dim: int, features: Optional[list] = None, input_dims: list = None, drop_feats: list = None, dropout: float = 0.2):
        """
        输入特征嵌入层
        参数:
            fea: 特征名称
            embed_dim: 嵌入维度
        """
        super().__init__()

        self.features = features
        self.drop_feats = drop_feats if drop_feats is not None else []
        self.embeddings = nn.ModuleDict()
        for fea, in_dim in zip(features, input_dims):
            if fea == 'ss_label':
                self.embeddings['ss_label'] = nn.Sequential(
                    nn.Embedding(9, embed_dim),  # 0为mask，1-8对应8个二级结构
                )
            elif 'label' in fea:
                # 获取对应边界
                fea_type = fea.split('_')[0]
                self.register_buffer(f'{fea_type}_boundaries', torch.tensor(eval(fea_type.upper() + '_DISCRETIZATION_BOUNDARIES')))
                boundaries = getattr(self, f'{fea_type}_boundaries')
                n_bins = len(boundaries) - 1  # 桶数=边界数-1

                self.embeddings[fea] = nn.Sequential(
                    nn.Linear(n_bins, embed_dim),  # RBF输出维度 = 桶数
                    Swish(),
                    nn.LayerNorm(embed_dim),
                )
            else:
                self.embeddings[fea] = nn.Sequential(
                    nn.Linear(in_dim, embed_dim),
                    Swish(),
                    nn.LayerNorm(embed_dim),
                    #nn.Linear(input_dim, embed_dim),
                )
            
            if fea in self.drop_feats:
                self.embeddings[fea].add_module('dropout', nn.Dropout(dropout))
    
    def forward(self, data: Data):
        feature_embs = []
        for fea in self.features:
            raw = getattr(data, fea)
            
            # 处理 ss_label (-1视为无效)
            if fea in ['ss_label', 'hbond_label']:
                emb = self.embeddings[fea](raw + 1)  # 将-1转换为0，其他值+1
            # 处理分桶标签
            elif 'label' in fea:
                boundaries = getattr(self, f'{fea.split("_")[0]}_boundaries')
                mask = ~torch.isnan(raw)
                raw_filled = raw.clone()
                raw_filled[~mask] = boundaries[0]  # nan替换成最小值
                rbf_enc = rbf(raw_filled, boundaries)
                emb = self.embeddings[fea](rbf_enc)
                emb[~mask] = 0.0
            else:
                emb = self.embeddings[fea](raw)
            feature_embs.append(emb)
        return feature_embs

