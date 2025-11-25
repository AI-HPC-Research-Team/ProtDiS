import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from src.models.layers import MLP, grad_reverse
from src.models.losses import info_nce_loss, redundancy_reduction_loss
from src.datasets.data_utils import (
    ASA_DISCRETIZATION_BOUNDARIES, asa_mean, asa_std,
    BFACTOR_DISCRETIZATION_BOUNDARIES, bfactor_mean, bfactor_std,
    LP_DISCRETIZATION_BOUNDARIES, lp_mean, lp_std,
    HYDRO_DISCRETIZATION_BOUNDARIES, hydro_mean, hydro_std,
    STABILITY_DISCRETIZATION_BOUNDARIES, stability_mean, stability_std,
    ENTROPY_DISCRETIZATION_BOUNDARIES, entropy_mean, entropy_std,
    ORC_DISCRETIZATION_BOUNDARIES, orc_mean, orc_std
)
from src.datasets.data_utils import ss_weights

class KON_base(nn.Module):
    def __init__(self):
        super(KON_base, self).__init__()
        self.ckns = None
        self.ksns = None

        # 对抗损失权重
        self.max_lambda_adv_global = 0
        self.max_lambda_adv_local = 0
        self.update_lambda(0)

    def update_lambda(self, scale: float):
        """
        更新对抗损失权重
        """
        self.lambda_adv_global = scale * self.max_lambda_adv_global
        self.lambda_adv_local = scale * self.max_lambda_adv_local

    def f_map(self, x):
        raise NotImplementedError
    
class KON(KON_base):
    """
    Knowledge Orthogonal Network for multitask prediction with InfoBottleneck on sk:
      1. 输入是结构编码器提取的复杂特征；
      2. 最小化公共特征的信息，同时最小化任务特定特征 sk 与输入 X 的互信息 (InfoBottleneck via KL loss).

    参数:
      d_model (int): 隐藏层维度。
      c_layers (list): 主知识网络的特征提取层数。
      k_layers (list): 知识特定分支的特征提取网络层数。
      lambda_adv (float): 对抗损失权重。
      lambda_mi (float): 互信息约束 InfoNCE 权重（已有）。
      lambda_kl (float): KL 信息瓶颈权重（新加）。
      mi_temp (float): InfoNCE 温度。
    """
    def __init__(self,
                 d_model: int,
                 c_layers: list,
                 k_layers: list,
                 features: list = None,
                 common_dims: list = [128, 128],
                 feature_dims: list = None,
                 discrete_labels: bool = True,
                 class_weight: bool = True,
                 feature_restore: bool = False,
                 lambda_adv: float = 0,
                 lambda_kl: float = 0,
                 lambda_mi: float = 0,
                 mi_temp: float = 0.2):
        super(KON, self).__init__()
        # adversarial weights
        self.max_lambda_adv_global = self.max_lambda_adv_local = lambda_adv
        self.lambda_mi, self.mi_temp = lambda_mi, mi_temp
        self.lambda_kl = lambda_kl
        self.features = features or ['ss', 'asa', 'bfactor', 'lp', 'hydro', 'stability', 'entropy', 'orc']
        if feature_dims is None:
            feature_dims = [common_dims[1]] * len(self.features)
        assert len(self.features) == len(feature_dims), "features and feature_dims must have the same length"
        self.common_dims = common_dims
        self.feature_dims = feature_dims
        self.discrete_labels = discrete_labels
        self.feature_restore = feature_restore

        # Common knowledge network
        self.ckns = nn.ModuleList()
        for i in range(2):
            self.ckns.append(
                nn.Sequential(
                    nn.Linear(128 if i==0 else self.common_dims[0], d_model),
                    *[MLP(d_model) for _ in range(c_layers[i])],
                    nn.Linear(d_model, self.common_dims[i]),
                    nn.LayerNorm(self.common_dims[i]),
                ) if c_layers[i] > 0 else None
            )

        # Knowledge specific network
        self.ksns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i, fea in enumerate(self.features):
            # feature extractor for each knowledge type
            self.ksns.append(
                nn.Sequential(
                    nn.Linear(self.common_dims[0], d_model),
                    *[MLP(d_model) for _ in range(k_layers[i])],
                    nn.Linear(d_model, self.feature_dims[i]),
                    #nn.LayerNorm(self.feature_dims[i]),
                )
            )
            # head that maps sk_i to output embedding for each task
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(self.common_dims[1] + self.feature_dims[i], d_model),
                    MLP(d_model),
                    MLP(d_model),
                )
            )

        if self.feature_restore:
            self.restore_net = nn.Sequential(
                nn.Linear(self.common_dims[1], d_model),
                #nn.Linear(self.common_dims[1] + sum(self.feature_dims), d_model),
                MLP(d_model),
                MLP(d_model),
                nn.Linear(d_model, 128),
            )
            self.regression_loss = nn.L1Loss()

        # task specific heads
        head_dims = {}
        if 'ss' in self.features:
            head_dims['ss'] = 8
            self.ss_head = nn.Linear(d_model, head_dims['ss'])
            self.ss_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ss_weights).float(), ignore_index=-1) if class_weight else nn.CrossEntropyLoss(ignore_index=-1)
        if 'bfactor' in self.features:
            head_dims['bfactor'] = len(BFACTOR_DISCRETIZATION_BOUNDARIES) - 2
            self.bfactor_head = nn.Linear(d_model, head_dims['bfactor'])
            self.bfactor_criterion = nn.CrossEntropyLoss(ignore_index=head_dims['bfactor'])

        for i, fea in enumerate(['asa', 'lp', 'hydro', 'stability', 'entropy', 'orc']):
            if fea in self.features:
                head_dims[fea] = len(eval(f"{fea.upper()}_DISCRETIZATION_BOUNDARIES")) - 1
                head = nn.Linear(d_model, head_dims[fea])
                setattr(self, f"{fea}_head", head)
                criterion = nn.CrossEntropyLoss()
                setattr(self, f"{fea}_criterion", criterion)

        for i, fea in enumerate(['bfactor', 'asa', 'lp', 'hydro', 'stability', 'entropy', 'orc']):
            if fea in self.features:
                setattr(self, f'{fea}_reg_head', nn.Linear(d_model, 1))
                setattr(self, f'{fea}_reg_loss', nn.L1Loss())

        # mutual infomation constraint
        if self.lambda_mi > 0:
            self.proj_enc = nn.Sequential(
                nn.Linear(128, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 128),
            )
            self.proj_ck = nn.Sequential(
                nn.Linear(self.common_dims[1], d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 128),
            )
            self.proj_sk = nn.ModuleList()
            for i, fea in enumerate(self.features):
                self.proj_sk.append(
                    nn.Sequential(
                        nn.Linear(self.feature_dims[i], d_model),
                        nn.ReLU(inplace=True),
                        nn.Linear(d_model, 128),
                    )
                )

        if self.lambda_kl > 0:
            self.fcs2 = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.feature_dims[i], d_model),
                    MLP(d_model),
                    MLP(d_model),
                    nn.Linear(d_model, head_dims[fea] if self.discrete_labels or fea == 'ss' else 1)
                ) for i, fea in enumerate(self.features)
            )

        # adversarial branches
        if self.max_lambda_adv_local > 0:
            self.adv_branch = nn.ModuleDict()
            for fea in self.features:
                self.adv_branch[fea] = nn.Sequential(
                    nn.Linear(self.common_dims[1], d_model),
                    MLP(d_model),
                    MLP(d_model),
                    nn.Linear(d_model, head_dims[fea])
                )

        #############################################################################################
        # 可以对 ck2 与 sk 之间的信息做判别器，让判别器去分辨是来自 PUBLIC 还是 PRIVATE，进一步“分割”两者。
        #############################################################################################

    def f_map(self, x):
        # produce structural features for each task
        ck1 = self.ckns[0](x) if self.ckns[0] else x
        ck2 = self.ckns[1](ck1) if self.ckns[1] else torch.zeros_like(ck1, device=ck1.device)
        ck2_adv = self.ckns[1](ck1.detach()) if self.ckns[1] else torch.zeros_like(ck1, device=ck1.device)
        sk = [self.ksns[i](ck1) for i, fea in enumerate(self.features)]
        outs = [self.fcs[i](torch.cat([ck2, sk[i]], dim=-1)) for i, fea in enumerate(self.features)]
        return {
            'ck': (ck1, ck2),
            'sk': tuple(sk),
            'ck2_adv': ck2_adv,
            'outputs': tuple(outs)
        }

    def forward(self, data):
        device = next(self.parameters()).device
        x_enc = data.s_enc # (num_nodes, 128)
        # forward map
        outputs = self.f_map(x_enc)
        ck1, ck2 = outputs['ck']
        sk_list = list(outputs['sk'])
        
        loss_dict = {}

        # ---------------------------
        #  knowledge specific losses
        # ---------------------------
        raw_labels = {fea: getattr(data, f"{fea}_label") for fea in self.features}
        cont_labels, disc_labels = {}, {}
        for i, fea in enumerate(self.features):
            if fea != 'ss':
                _mean = eval(f"{fea}_mean")
                _std = eval(f"{fea}_std")
                cont_labels[fea] = (raw_labels[fea] - _mean) / (_std + 1e-6)  # 标准化

                boundaries = eval(f"{fea.upper()}_DISCRETIZATION_BOUNDARIES")
                disc_labels[fea] = torch.bucketize(
                    raw_labels[fea],
                    torch.tensor(boundaries, dtype=torch.float, device=device)[1:-1],
                    right=True,
                )
        disc_labels['ss'] = raw_labels['ss']

        masks = {fea: ~torch.isnan(raw_labels[fea]) for fea in self.features if fea != 'ss'}
        masks['bfactor'] = (raw_labels['bfactor'] < 150)  # bfactor有效值范围[0,150)

        for i, fea in enumerate(self.features):
            fea_encodings = outputs['outputs'][i]

            # secondary structure (cross-entropy)
            if fea == 'ss':
                ss_label = disc_labels['ss']
                loss_dict['ss_loss'] = self.ss_criterion(self.ss_head(fea_encodings), ss_label.long())                
            # Other features (cross-entropy) with NaN mask
            else:
                criterion = getattr(self, f'{fea}_criterion') if self.discrete_labels else getattr(self, f'{fea}_reg_loss')
                task_head = getattr(self, f'{fea}_head') if self.discrete_labels else getattr(self, f'{fea}_reg_head')

                valid_mask = masks[fea]
                if valid_mask.any():
                    if self.discrete_labels:
                        loss_dict[f'{fea}_loss'] = criterion(task_head(fea_encodings[valid_mask]), disc_labels[fea][valid_mask])
                    else:
                        loss_dict[f'{fea}_loss'] = criterion(task_head(fea_encodings[valid_mask]).squeeze(-1), cont_labels[fea][valid_mask])
                else:
                    loss_dict[f'{fea}_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
            
                if fea == 'bfactor':
                    # 在损失计算后插入检查
                    if torch.isnan(loss_dict['bfactor_loss']).any() or torch.isinf(loss_dict['bfactor_loss']).any():
                        #print(f"NaN/Inf detected in bfactor loss! set loss to zero.")
                        loss_dict['bfactor_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

        # restore features
        if self.feature_restore:
            restored = self.restore_net(ck2)  # (N, 128)
            #restored = self.restore_net(torch.cat([ck2] + sk_list, dim=-1))  # (N, 128)
            loss_dict['restore_loss'] = self.regression_loss(restored, x_enc)

        # -------------------------
        #  adversarial losses
        # -------------------------
        if self.max_lambda_adv_local > 0:
            fea_list = list(self.adv_branch.keys())
            adv_fea = grad_reverse(outputs['ck2_adv'], self.lambda_adv_local)
            for fea in fea_list:
                if fea == 'ss':
                    ss_label = disc_labels['ss']
                    adv_ss_logits = self.adv_branch['ss'](adv_fea)
                    loss_dict['adv_ss_loss'] = self.ss_criterion(adv_ss_logits, ss_label.long())
                else:
                    valid_mask = masks[fea]
                    if valid_mask.any():
                        adv_logits = self.adv_branch[fea](adv_fea)
                        criterion = getattr(self, f'{fea}_criterion')
                        loss_dict[f'adv_{fea}_loss'] = criterion(adv_logits[valid_mask], disc_labels[fea][valid_mask])

        # -------------------------
        #  mutual information loss
        # -------------------------
        if self.lambda_mi > 0:
            z_mi = self.proj_enc(x_enc)  # (N, 128)
            z_ck2 = self.proj_ck(ck2)

            mi_ck = info_nce_loss(z_ck2, z_mi, self.mi_temp)
            loss_dict['mi_ck_loss'] = self.lambda_mi * mi_ck
            for i, fea in enumerate(self.features):
                sk = sk_list[i]
                z_sk = self.proj_sk[i](sk)  # (N, 128)
                mi_sk = info_nce_loss(z_sk, z_mi, self.mi_temp)  # 对原始特征和sk之间的互信息进行约束
                #mi_sk = info_nce_loss(z_sk, z_ck2, self.mi_temp)  # 对ck2和sk之间的互信息进行约束
                loss_dict[f'mi_{fea}_loss'] = - self.lambda_mi * mi_sk

        # ------------------------------------
        #  InfoBottleneck (KL) on task features
        # ------------------------------------
        if self.lambda_kl > 0:
            eps = 1e-6
            for i, fea in enumerate(self.features):
                z = sk_list[i]  # shape [N, D]
                mu = z.mean(dim=0)
                var = z.var(dim=0, unbiased=False) + eps
                kl = -0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
                loss_dict[f'kl_{fea}_loss'] = self.lambda_kl * kl

            loss_dict['decorrelation_loss'] = redundancy_reduction_loss(sk_list, lamb_var=0.1, lamb_cov=0.005)

            for i, fea in enumerate(self.features):
                if fea == 'ss':
                    ss_label = disc_labels['ss']
                    loss_dict['sk_ss_loss'] = self.ss_criterion(self.fcs2[i](sk_list[i]), ss_label.long())
                else:
                    valid_mask = masks[fea]
                    if valid_mask.any():
                        if self.discrete_labels:
                            criterion = getattr(self, f'{fea}_criterion')
                            loss_dict[f'sk_{fea}_loss'] = criterion(self.fcs2[i](sk_list[i])[valid_mask], disc_labels[fea][valid_mask])
                        else:
                            criterion = getattr(self, f'{fea}_reg_loss')
                            loss_dict[f'sk_{fea}_loss'] = criterion(self.fcs2[i](sk_list[i])[valid_mask].squeeze(-1), cont_labels[fea][valid_mask])

                    if fea == 'bfactor':
                        # 在损失计算后插入检查
                        if torch.isnan(loss_dict['sk_bfactor_loss']).any() or torch.isinf(loss_dict['sk_bfactor_loss']).any():
                            #print(f"NaN/Inf detected in bfactor loss! set loss to zero.")
                            loss_dict['sk_bfactor_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

        # total
        loss_dict['total_loss'] = sum(loss_dict.values())
        return loss_dict

class HierKON(KON_base):
    def __init__(self):
        raise NotImplementedError("HierKON is not implemented yet.")
