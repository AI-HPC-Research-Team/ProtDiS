import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from esm.layers.transformer_stack import TransformerStack
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

class TransformerBlock(nn.Module):
    def __init__(self,
                 n_heads: 8,
                 v_heads: 8,
                 n_layers: 16,
                 d_model: int = 256,) -> nn.Module:
        super().__init__()

        self.blocks = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
        )

    def forward(self, x: torch.Tensor, data: Data):
        structure_coords = data.pos.unsqueeze(0)[..., :3, :]  # (b, l, 3, 3)
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        out, embedding, _ = self.blocks(
            x.unsqueeze(0), data.batch.unsqueeze(0), affine, affine_mask
        ) # (1, num_nodes, d_model)

        out = out.squeeze(0)
        return out
