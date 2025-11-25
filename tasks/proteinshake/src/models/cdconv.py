import numpy as np
import math
import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_sum
from torch_sparse import SparseTensor, set_diag
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch_geometric.transforms as T
from torch_geometric.nn import fps, global_mean_pool, global_add_pool, global_max_pool, radius
from torch_geometric.nn.pool import avg_pool, max_pool

def kaiming_uniform(tensor, size):
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class WeightNet(nn.Module):
    def __init__(self, l: int, kernel_channels: list):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        for i, channels in enumerate(kernel_channels):
            if i == 0:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, 3 + 3 + 1, channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
            else:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, kernel_channels[i-1], channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))

        self.relu = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        for i, channels in enumerate(self.kernel_channels):
            if i == 0:
                kaiming_uniform(self.Ws[0].data, size=[self.l, 3 + 3 + 1, channels])
            else:
                kaiming_uniform(self.Ws[i].data, size=[self.l, self.kernel_channels[i-1], channels])
            self.bs[i].data.fill_(0.0)

    def forward(self, input, idx):
        for i in range(len(self.kernel_channels)):
            W = torch.index_select(self.Ws[i], 0, idx)
            b = torch.index_select(self.bs[i], 0, idx)
            if i == 0:
                weight = self.relu(torch.bmm(input.unsqueeze(1), W).squeeze(1) + b)
            else:
                weight = self.relu(torch.bmm(weight.unsqueeze(1), W).squeeze(1) + b)

        return weight

class CDConv(MessagePassing):
    def __init__(self, r: float, l: float, kernel_channels: list, in_channels: int, out_channels: int, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'sum')
        super().__init__(**kwargs)
        self.r = r
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.WeightNet = WeightNet(l, kernel_channels)
        self.W = torch.nn.Parameter(torch.empty(kernel_channels[-1] * in_channels, out_channels))

        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform(self.W.data, size=[self.kernel_channels * self.in_channels, self.out_channels])

    def forward(self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor) -> Tensor:
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=9999)
        edge_index = torch.stack([col, row], dim=0)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos.size(0), pos.size(0)))

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x, None), pos=(pos, pos), seq=(seq, seq), ori=(ori.reshape((-1, 9)), ori.reshape((-1, 9))), size=None)
        out = torch.matmul(out, self.W)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, seq_i: Tensor, seq_j: Tensor, ori_i: Tensor, ori_j: Tensor) -> Tensor:
        # orientation
        pos = pos_j - pos_i
        distance = torch.norm(input=pos, p=2, dim=-1, keepdim=True)
        pos /= (distance + 1e-9)

        pos = torch.matmul(ori_i.reshape((-1, 3, 3)), pos.unsqueeze(2)).squeeze(2)
        ori = torch.sum(input=ori_i.reshape((-1, 3, 3)) * ori_j.reshape((-1, 3, 3)), dim=2, keepdim=False)

        #
        normed_distance = distance / self.r

        seq = seq_j - seq_i
        s = self.l//2
        seq = torch.clamp(input=seq, min=-s, max=s)
        seq_idx = (seq + s).squeeze(1).to(torch.int64)
        normed_length = torch.abs(seq) / s

        # generated kernel weight: PointConv or PSTNet
        delta = torch.cat([pos, ori, distance], dim=1)
        kernel_weight = self.WeightNet(delta, seq_idx)

        # smooth: IEConv II
        smooth = 0.5 - torch.tanh(normed_distance*normed_length*16.0 - 14.0)*0.5

        # convolution
        msg = torch.matmul((kernel_weight*smooth).unsqueeze(2), x_j.unsqueeze(1))

        msg = msg.reshape((-1, msg.size(1)*msg.size(2)))

        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(r={self.r}, '
                f'l={self.l},'
                f'kernel_channels={self.kernel_channels},'
                f'in_channels={self.in_channels},'
                f'out_channels={self.out_channels})')

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_max(src=x, index=idx, dim=0)[0]
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch, idx

class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch, idx
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, batch_norm=True, dropout=0.0, bias=False):
        super().__init__()
        # 上采样模块：融合跳跃连接和上采样特征
        self.up_linear = nn.Linear(in_channels, out_channels, bias=bias)
        if skip_channels and skip_channels > 0:
            # 如果有跳跃连接通道，则使用它们
            self.skip_linear = nn.Linear(skip_channels, out_channels, bias=bias)
        else:
            self.skip_linear = None
        self.fuse = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
    def forward(self, x, skip, idx):
        # 上采样：通过索引扩展特征
        upsampled = x[idx]
        # 处理跳跃连接
        skip_proj = self.skip_linear(skip) if self.skip_linear else 0
        # 融合特征
        fused = self.up_linear(upsampled) + skip_proj
        return self.fuse(fused)
    
class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list,
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out
    


class CDConvEncoder(nn.Module):
    def __init__(self,
                 geometric_radii: List,
                 sequential_kernel_size: float,
                 kernel_channels: float,
                 channels: List,
                 base_width: float = 16.0,
                 embed_dim: int = 16,
                 d_model: int = 256,
                 pooling: str = 'mean',
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False) -> nn.Module:
        super().__init__()
        assert len(geometric_radii) == len(channels), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.local_pool = AvgPooling() if pooling == 'mean' else MaxPooling()

        layers = []
        in_channels = embed_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = [kernel_channels],
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = [kernel_channels],
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)
        self.global_pool = eval(f'global_{pooling}_pool') if pooling else None
        self.fc_out = MLP(in_channels=channels[-1],
                        mid_channels=max(channels[-1], d_model),
                        out_channels=d_model,
                        batch_norm=batch_norm,
                        dropout=dropout)

    def forward(self, x: torch.Tensor, data: Data):
        pos, aid, ori, batch = data.pos, data.aid, data.ori, data.batch

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, aid, ori, batch)
            if self.global_pool:
                if i == len(self.layers) - 1:
                    x = self.global_pool(x, batch)
                elif i % 2 == 1:
                    x, pos, aid, ori, batch, _ = self.local_pool(x, pos, aid, ori, batch)
        out = self.fc_out(x)
        return out


class CDConvUNet(nn.Module):
    def __init__(self,
                 geometric_radii: List,
                 sequential_kernel_size: float,
                 kernel_channels: float,
                 channels: List,
                 base_width: float = 16.0,
                 d_model: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False) -> nn.Module:
        super().__init__()
        assert len(geometric_radii) == len(channels), "geometric_radii and channels should have same length!"
        
        self.local_pool = AvgPooling()
        
        # 创建编码器块
        self.enc_blocks = nn.ModuleList()
        in_channels = d_model
        for i, radius in enumerate(geometric_radii):
            # 添加两个连续的基本块
            self.enc_blocks.append(BasicBlock(r=radius, 
                                              l=sequential_kernel_size,
                                              kernel_channels=[kernel_channels],
                                              in_channels=in_channels,
                                              out_channels=channels[i],
                                              base_width=base_width,
                                              batch_norm=batch_norm,  # 应该改为layernorm
                                              dropout=dropout,
                                              bias=bias))
            self.enc_blocks.append(BasicBlock(r=radius, 
                                              l=sequential_kernel_size,
                                              kernel_channels=[kernel_channels],
                                              in_channels=channels[i],
                                              out_channels=channels[i],
                                              base_width=base_width,
                                              batch_norm=batch_norm,
                                              dropout=dropout,
                                              bias=bias))
            in_channels = channels[i]

        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=channels[-1],
            num_heads=4,
            dropout=dropout
        )
        
        # 解码器路径
        self.dec_blocks = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        
        # 创建解码器块（反向顺序）
        for i in range(len(geometric_radii)-2, -1, -1):
            # 添加上采样层
            self.upsample_ops.append(UpSampling(
                in_channels=channels[i+1],
                out_channels=channels[i],
                skip_channels=channels[i],
                batch_norm=batch_norm,
                dropout=dropout,
                bias=bias
            ))
            # 添加两个连续的基本块
            self.dec_blocks.append(BasicBlock(
                r=geometric_radii[i], 
                l=sequential_kernel_size,
                kernel_channels=[kernel_channels],
                in_channels=channels[i],
                out_channels=channels[i],
                base_width=base_width,
                batch_norm=batch_norm,
                dropout=dropout,
                bias=bias
            ))
            self.dec_blocks.append(BasicBlock(
                r=geometric_radii[i], 
                l=sequential_kernel_size,
                kernel_channels=[kernel_channels],
                in_channels=channels[i],
                out_channels=channels[i],
                base_width=base_width,
                batch_norm=batch_norm,
                dropout=dropout,
                bias=bias
            ))
        
        # 最终输出层
        # self.fc_out = MLP(in_channels=channels[0],
        #                 mid_channels=max(channels[0], d_model),
        #                 out_channels=d_model,
        #                 batch_norm=batch_norm,
        #                 dropout=dropout)
        self.fc_out = nn.Linear(channels[0], d_model, bias=bias)
        
    def forward(self, x: torch.Tensor, data: Data):
        pos, aid, ori, batch = data.pos, data.aid, data.ori, data.batch
        
        # 存储跳跃连接和池化信息
        skip_connections = []
        pool_indices = []
        pool_infos = []
        
        # 编码器路径
        for i in range(len(self.enc_blocks)):
            x = self.enc_blocks[i](x, pos, aid, ori, batch)
            
            # 在每两个块后进行下采样（最后一个块除外）
            if (i % 2 == 1) and (i < len(self.enc_blocks) - 1):
                # 保存跳跃连接
                skip_connections.append(x)
                pool_infos.append((pos, aid, ori, batch))
                
                # 执行池化
                x, pos, aid, ori, batch, idx = self.local_pool(x, pos, aid, ori, batch)
                pool_indices.append(idx)
        
        # 瓶颈层（编码器最深层）
        bottleneck = x.unsqueeze(0)  # [1, N, C]
        bottleneck, _ = self.bottleneck_attn(bottleneck, bottleneck, bottleneck)
        x = bottleneck.squeeze(0)
        
        # 解码器路径
        for i in range(len(self.upsample_ops)):
            # 上采样并融合跳跃连接
            skip = skip_connections.pop()
            idx = pool_indices.pop()
            x = self.upsample_ops[i](x, skip, idx)
            
            # 通过两个基本块
            pos, aid, ori, batch = pool_infos.pop()
            x = self.dec_blocks[2*i](x, pos, aid, ori, batch)
            x = self.dec_blocks[2*i+1](x, pos, aid, ori, batch)
        
        # 最终输出变换
        out = self.fc_out(x)
        return out


