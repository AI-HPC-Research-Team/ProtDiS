import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    """
    支持多种融合策略的特征融合模块
    - fusion_method: 'gate' / 'attention' / 'gate_attention' / 'attention_gate' / 'concat'
    - s_dim: s_embeddings 的维度
    - gate_temp: softmax 温度（默认 1.0），可训练
    - feature_dropout: 在特征级别随机丢弃整个特征向量的概率（0~1）
    """
    def __init__(self, fusion_method, d_model, num_features, s_dim=128, 
                 attn_heads=4, attn_dropout=0.1, gate_temp=1.0):
        super().__init__()
        self.fusion_method = fusion_method
        self.num_features = num_features
        self.d_model = d_model
        self.s_dim = s_dim
        self.gate_temp = gate_temp

        self.layer_norm = nn.LayerNorm(d_model)
        if 'gate' in fusion_method:
            self.gating_net = nn.Sequential(
                nn.Linear(s_dim, max(s_dim, 64)),
                nn.ReLU(inplace=True),
                nn.Linear(max(s_dim, 64), num_features)
            )
            # 统一用一个可学习参数存温度，但可选择是否训练
            self.gate_temp = nn.Parameter(torch.tensor(float(gate_temp)))
            self.gate_temp.requires_grad = False  # 默认不训练温度参数
            self.softmax = nn.Softmax(dim=-1)  # 对最后一个维度（num_features）softmax
        if 'attention' in fusion_method:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                        num_heads=attn_heads,
                                                        dropout=attn_dropout,
                                                        batch_first=True)
        if fusion_method == 'concat':
            self.fuse = self._concat_fusion
            self.fc_net = nn.Linear(num_features * d_model, d_model)
        elif fusion_method == 'gate':
            self.fuse = self._gate_fusion
        elif fusion_method == 'attention':
            self.fuse = self._attention_fusion
        elif fusion_method == 'gate_attention':
            self.fuse = self._gate_attention_fusion
        elif fusion_method == 'attention_gate':
            self.fuse = self._attention_gate_fusion
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self, features, **kwargs):
        assert len(features) == self.num_features, "features length mismatch num_features"
        return self.fuse(features=features, **kwargs)
    
    def _concat_fusion(self, features, **kwargs):
        """简单拼接后线性变换"""
        fused = torch.cat(features, dim=-1)  # (batch, N*d_model)
        fused = self.fc_net(fused)  # (batch, d_model)
        return fused, None

    def _gate_fusion(self, features, s_embeddings):
        """基于全局s_embeddings (batch, d_model) 计算num_features维度的权重，并融合features"""
        raw = self.gating_net(s_embeddings)  # (batch, num_features)
        temp = torch.clamp(self.gate_temp, min=1e-6)
        weights = self.softmax(raw / temp)  # (B, N)
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused += feat * weights[:, i].unsqueeze(-1)  # (batch, d_model)

        residual = torch.stack(features, dim=1).mean(dim=1)  # (batch, d_model)
        fused = self.layer_norm(fused + residual)
        return fused, weights
    
    def _attention_fusion(self, features, **kwargs):
        """多头注意力融合 (self-attention)"""
        feats_cat = torch.stack(features, dim=1)  # (B, N, d_model)
        attn_out, attn_weights = self.multihead_attn(feats_cat, feats_cat, feats_cat)  # (B, N, d_model)
        fused = attn_out.mean(dim=1)  # (B, d_model)
        fused = self.layer_norm(fused + feats_cat.mean(dim=1))
        return fused, attn_weights

    def _gate_attention_fusion(self, features, s_embeddings):
        # 1. Gate 权重
        raw = self.gating_net(s_embeddings)  # (B, N)
        temp = torch.clamp(self.gate_temp, min=1e-6)
        weights = self.softmax(raw / temp)  # (B, N)

        # 2. 每个特征缩放
        gated_feats = torch.stack([
            feat * weights[:, i].unsqueeze(-1) for i, feat in enumerate(features)
        ], dim=1)  # (B, N, d_model)

        # 3. Query 用 gate 融合后的全局特征
        query = gated_feats.sum(dim=1, keepdim=True)  # (B, 1, d_model)

        # 4. Attention
        attn_out, _ = self.multihead_attn(query, gated_feats, gated_feats)  # (B, 1, d_model)
        attn_out = attn_out.squeeze(1)

        # 5. 残差 & LN
        residual = torch.stack(features, dim=1).mean(dim=1)
        fused = self.layer_norm(attn_out + residual)

        return fused, weights

    def _attention_gate_fusion(self, features, s_embeddings):
        # 1. 多头注意力融合 (self-attention)
        feats_cat = torch.stack(features, dim=1)  # (B, N, d_model)
        attn_out, _ = self.multihead_attn(feats_cat, feats_cat, feats_cat)  # (B, N, d_model)

        # 2. Gate 加权（基于 s_embeddings）
        raw = self.gating_net(s_embeddings)  # (B, num_features)
        temp = torch.clamp(self.gate_temp, min=1e-6)
        weights = self.softmax(raw / temp)  # (B, N)

        fused = torch.zeros_like(features[0])
        for i in range(self.num_features):
            fused += attn_out[:, i, :] * weights[:, i].unsqueeze(-1)  # (B, d_model)

        # 3. 残差 + LayerNorm
        residual = feats_cat.mean(dim=1)  # (B, d_model)
        fused = self.layer_norm(fused + residual)

        return fused, weights

class Aggregator(nn.Module):
    def __init__(self, embed_dim=256, aggregation='concat', normalize=False):
        super().__init__()
        self.aggregation = aggregation
        self.normalize = normalize

        if aggregation == 'concat':
            self.aggregator = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )
        elif aggregation == 'dot' or aggregation == 'sum':
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, x1, x2):
        if self.normalize:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        if self.aggregation == 'concat':
            x = torch.cat((x1, x2), dim=-1)
        elif self.aggregation == 'dot':
            x = x1 * x2
        elif self.aggregation == 'sum':
            x = x1 + x2
        return self.aggregator(x)
