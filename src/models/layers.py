import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
#import torch_npu
from torch_geometric.nn import global_mean_pool
from esm.layers.transformer_stack import TransformerStack
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
    
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)

class MLP(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 expansion_ratio: float = 2.0,
                 layer_norm: bool = True,
                 bias: bool = True) -> nn.Module:
        super(MLP, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        hidden_channels = swiglu_correction_fn(expansion_ratio, in_channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channels) if layer_norm else nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, hidden_channels * 2, bias=bias),
            SwiGLU(),
            nn.Linear(hidden_channels, out_channels, bias=bias),)
        if out_channels != in_channels:
            self.short_connect = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            self.short_connect = nn.Identity()
        #self.init_weights('eye')
        self.device = next(self.parameters()).device

    def init_weights(self, mode):
        if mode == 'eye':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if m.weight.shape[0] == m.weight.shape[1]:  # 仅对方阵使用 eye
                        nn.init.eye_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)  # 非方阵使用 Xavier
        elif mode == 'xav':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.short_connect(x) + self.ffn(x)

class TransformerFuse(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, d_model: int, n_heads: int, v_heads: int, n_layers: int):
        super(TransformerFuse, self).__init__()
        self.fc_in = nn.Linear(input_dims, d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, v_heads, n_layers, mask_and_zero_frameless=True
        )
        self.fc_out = nn.Linear(d_model, output_dims)
    
    def forward(self, x: torch.Tensor, coordinates: torch.Tensor, sequence_id: torch.Tensor):
        if len(x.shape) == 2: x = x.unsqueeze(0)
        if len(coordinates.shape) == 3: coordinates = coordinates.unsqueeze(0)
        
        structure_coords = coordinates[..., :3, :] # (1, L, 3, 3)
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        x = self.fc_in(x)
        x, embedding, _ = self.transformer(x, sequence_id, affine, affine_mask)
        x = self.fc_out(x)
        x = global_mean_pool(x.squeeze(0), sequence_id)
        return x
    

