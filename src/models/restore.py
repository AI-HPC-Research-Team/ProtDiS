import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import MLP
from src.datasets.data_utils import (
    ASA_DISCRETIZATION_BOUNDARIES, 
    BFACTOR_DISCRETIZATION_BOUNDARIES, 
    LP_DISCRETIZATION_BOUNDARIES, 
    HYDRO_DISCRETIZATION_BOUNDARIES, 
    STABILITY_DISCRETIZATION_BOUNDARIES, 
    ENTROPY_DISCRETIZATION_BOUNDARIES,
    ORC_DISCRETIZATION_BOUNDARIES
)
from src.datasets.data_utils import ss_weights
    
class RestoreNetwork(nn.Module):
    """
    Restore the knowledge labels from the embeddings.
      
    参数:
      d_model (int): 隐藏层维度。
      layers (int): 全连接网络层数。
    """
    def __init__(self,
                 d_model: int = 128,
                 layers: int = 5,
                 input_encs: list = None,
                 input_dims: list = None,
                 restore_label: str = 's_enc',
                 class_weight: bool = True,):
        super().__init__()
        self.input_encs = input_encs or ['ck', 'ss', 'asa', 'bfactor', 'lp', 'hydro', 'stability', 'entropy', 'orc']
        if input_dims is None:
            input_dims = [128] * len(self.input_encs)
        assert len(self.input_encs) == len(input_dims), "input_encs and input_dims must have the same length"
        self.restore_label = restore_label

        # 每个特征先映射到同一维度 d_model
        self.feature_maps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            ) for dim in input_dims
        ])

        layer_list = [
            nn.Linear(len(input_dims) * d_model, d_model),
            *[MLP(d_model) for _ in range(layers)]
        ]

        if restore_label == 's_enc':
            layer_list.append(nn.Linear(d_model, 128))
            # self.loss = nn.L1Loss()
            self.temperature = 0.1
        elif restore_label == 'ss':
            layer_list.append(nn.Linear(d_model, 8))
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(ss_weights).float(), ignore_index=-1) if class_weight else nn.CrossEntropyLoss(ignore_index=-1)
        elif restore_label == 'bfactor':
            layer_list.append(nn.Linear(d_model, len(BFACTOR_DISCRETIZATION_BOUNDARIES) - 2))
            self.loss = nn.CrossEntropyLoss(ignore_index=len(BFACTOR_DISCRETIZATION_BOUNDARIES) - 2)
        elif restore_label in ['asa', 'lp', 'hydro', 'stability', 'entropy', 'orc']:
            layer_list.append(nn.Linear(d_model, len(eval(f"{restore_label.upper()}_DISCRETIZATION_BOUNDARIES")) - 1))
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown restore_label: {restore_label}")
        self.restore_net = nn.Sequential(*layer_list)

    def forward(self, data):
        """
        input:
            dict including all knowledge labels and protein structure encodings
        return:
            loss dict
        """
        device = next(self.parameters()).device

        # 先对每个特征单独映射和归一化
        processed = []
        for enc, fmap in zip(self.input_encs, self.feature_maps):
            x = getattr(data, enc)
            x = fmap(x)
            processed.append(x)
            
        x_enc = torch.cat(processed, dim=-1)

        # forward map
        outputs = self.restore_net(x_enc)
        loss_dict = {}
        
        # ----------------
        #  compute losses
        # ----------------
        if self.restore_label == 's_enc':
            # loss_dict['restore_loss'] = self.loss(outputs, data.s_enc)
            # normalize embeddings
            outputs = F.normalize(outputs, dim=-1)  # [B, 128]
            targets = F.normalize(data.s_enc, dim=-1)  # [B, 128]

            # cosine similarity matrix
            logits = torch.matmul(outputs, targets.T) / self.temperature  # [B, B]
            labels = torch.arange(logits.size(0), device=device)

            loss_dict['infoNCE_loss'] = F.cross_entropy(logits, labels)

        elif self.restore_label == 'ss':
            loss_dict['ss_loss'] = self.loss(outputs, data.ss_label.long())

        else:
            raw_labels = getattr(data, f"{self.restore_label}_label")
            valid_mask = ~torch.isnan(raw_labels)
            if valid_mask.any():
                boundaries = eval(f"{self.restore_label.upper()}_DISCRETIZATION_BOUNDARIES")
                disc_labels = torch.bucketize(
                    raw_labels,
                    torch.tensor(boundaries, dtype=torch.float, device=device)[1:-1]
                )
                loss_dict[f'{self.restore_label}_loss'] = self.loss(outputs[valid_mask], disc_labels[valid_mask])
            else:
                loss_dict[f'{self.restore_label}_loss'] = torch.tensor(0.0, device=device, requires_grad=True)    

        # total
        loss_dict['total_loss'] = sum(loss_dict.values())
        return loss_dict

