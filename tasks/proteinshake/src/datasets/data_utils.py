import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i, acid in enumerate(aa):
    aa_to_id[acid] = i

SS_LIST = ['H: alpha helix', 'B: beta bridge', 'E: extended strand', 'G: 3_10 helix', 'I: π-helix', 'T: turn', 'S: bend', 'C: coil']
SS_LETTERS = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']
SS_MAP = {c:i for i,c in enumerate(SS_LETTERS)}
SS_MAP['-'] = SS_MAP['C']

def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)

def fuse_data(esm_file: str, k_file: str, resolution: str = 'residue', tsv_file: str = None, res_annos=None, chain_id=None):
    """
    Fuse ESM and K-encodings at residue or atom resolution.
    """
    if esm_file.endswith('.pth'):
        esm_encodings = torch.load(esm_file, map_location='cpu')
        esm_encodings['structure_encodings'] = esm_encodings['structure_encodings'].squeeze(0).numpy()
    else:
        esm_encodings = np.load(esm_file, allow_pickle=True)
    k_encodings = np.load(k_file, allow_pickle=True)

    data_df = pd.DataFrame({
        'chain_id': esm_encodings['chain_id'].tolist(),
        'res_id': esm_encodings['residue_index'].tolist(),
        'ins_code': esm_encodings['insertion_code'].tolist(),
        'res': esm_encodings['residue_type'].tolist(),
        'coordinates': list(esm_encodings['coordinates']),
        'structure_encodings': list(esm_encodings['structure_encodings']),
        'ck_encodings': list(k_encodings['ck_encodings']),
        'ss_encodings': list(k_encodings['ss_encodings']),
        'asa_encodings': list(k_encodings['asa_encodings']),
        'bfactor_encodings': list(k_encodings['bfactor_encodings']),
        'lp_encodings': list(k_encodings['lp_encodings']),
        'hydro_encodings': list(k_encodings['hydro_encodings']),
        'stability_encodings': list(k_encodings['stability_encodings']),
        'entropy_encodings': list(k_encodings['entropy_encodings']),
        'orc_encodings': list(k_encodings['orc_encodings']),
    })
    if chain_id is not None:
        data_df = data_df[data_df['chain_id'] == chain_id]
        if len(data_df) == 0:
            return None  # no data for this chain_id

    data_df = data_df.drop_duplicates(subset=['chain_id', 'res_id', 'res'], keep='first')  # remove duplicates
    
    # 处理标签文件 (TSV)
    if tsv_file is not None:
        if os.path.exists(tsv_file):
            label_df = pd.read_csv(tsv_file, sep='\t')
            #label_df = label_df[label_df['Insertion Code'].isin(['-', ''])] # remove residues with insertion codes
            label_df = label_df.rename(columns={
                'Chain': 'chain_id',
                'Residue ID': 'res_id',
                'Residue': 'res',
                'bfactor/confidence': 'bfactor'
            })
            label_df = label_df.drop_duplicates(subset=['chain_id', 'res_id', 'res'], keep='first')  # remove duplicates
            label_df['chain_id'] = label_df['chain_id'].apply(lambda col: str(col).strip())
            
            # 合并标签数据
            data_df = data_df.merge(
                label_df,
                on=['chain_id', 'res_id', 'res'], 
                how='left', 
                sort=False, 
            )

    # 处理残基标注 (res_annos)
    if res_annos is not None:
        chain_ids = set(res_annos['chain_id'])
        if not all(chain_id in data_df['chain_id'].values for chain_id in chain_ids):
            #print(f"warning: Some chain_ids in res_annos are not present in esm_encodings: {set(chain_ids) - set(data_df['chain_id'])}, drop these chains from {esm_file}.")
            chain_ids = chain_ids.intersection(set(data_df['chain_id']))
        data_df = data_df[data_df['chain_id'].isin(chain_ids)]  # filter by chain_id

        # 创建标注DataFrame并去重
        anno_df = pd.DataFrame({
            'chain_id': res_annos['chain_id'],
            'res_id': res_annos['residue_number'],
            'res': res_annos['residue_type'],
            'binding_site': res_annos['binding_site']
        }).drop_duplicates(subset=['chain_id', 'res_id', 'res'], keep='first')
        
        # 合并到主数据 (只更新binding_site列)
        data_df = data_df.merge(
            anno_df,
            on=['chain_id', 'res_id', 'res'],
            how='left',
        )
        
        # fill NaN binding_site to -1
        if 'binding_site' in data_df.columns:
            data_df['binding_site'] = data_df['binding_site'].fillna(-1)

    # process different resolutions
    if resolution == 'residue':
        ca_coords = np.array([pos[1] for pos in data_df['coordinates']])
        valid_mask = ~np.isnan(ca_coords).any(axis=1) # request Ca coordinates
        
        data_df = data_df[valid_mask].copy()
        ca_coords = ca_coords[valid_mask]
        data_df['coordinates'] = list(ca_coords)
        
        center = np.mean(ca_coords, axis=0, keepdims=True) # center of mass
        rel_coords = ca_coords - center
        ori = orientation(rel_coords)
        ori = torch.tensor(ori, dtype=torch.float)  # [L, 3, 3]
    
    elif resolution == 'atom':
        ori = None
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    
    # 转换为返回字典
    res_ids = torch.tensor(data_df['res_id'].values, dtype=torch.long)
    result = {
        'chain_id': data_df['chain_id'].values,
        'res_id': res_ids,
        'sequence': torch.tensor([aa_to_id[aa] for aa in data_df['res']], dtype=torch.long),
        'coordinates': torch.tensor(np.vstack(data_df['coordinates'].values), dtype=torch.float),
        'ori': ori,
        'structure_encodings': torch.tensor(np.vstack(data_df['structure_encodings'].values)),

        'ck_encodings': torch.tensor(np.vstack(data_df['ck_encodings'].values)),
        'ss_encodings': torch.tensor(np.vstack(data_df['ss_encodings'].values)),
        'asa_encodings': torch.tensor(np.vstack(data_df['asa_encodings'].values)),
        'bfactor_encodings': torch.tensor(np.vstack(data_df['bfactor_encodings'].values)),
        'lp_encodings': torch.tensor(np.vstack(data_df['lp_encodings'].values)),
        'hydro_encodings': torch.tensor(np.vstack(data_df['hydro_encodings'].values)),
        'stability_encodings': torch.tensor(np.vstack(data_df['stability_encodings'].values)),
        'entropy_encodings': torch.tensor(np.vstack(data_df['entropy_encodings'].values)),
        'orc_encodings': torch.tensor(np.vstack(data_df['orc_encodings'].values)),

        'ss_label': torch.tensor([SS_MAP.get(x, -1) for x in data_df['ss']], dtype=torch.long) if 'ss' in data_df.columns else -torch.ones_like(res_ids, dtype=torch.long),
        'asa_label': torch.tensor(data_df['asa'].values, dtype=torch.float) if 'asa' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'bfactor_label': torch.tensor(data_df['bfactor'].values, dtype=torch.float) if 'bfactor' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'lp_label': torch.tensor(data_df['local_packing'].values, dtype=torch.float) if 'local_packing' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'hydro_label': torch.tensor(data_df['hydrophobicity'].values, dtype=torch.float) if 'hydrophobicity' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'stability_label': torch.tensor(data_df['stability'].values, dtype=torch.float) if 'stability' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'entropy_label': torch.tensor(data_df['contact_entropy'].values, dtype=torch.float) if 'contact_entropy' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),
        'orc_label': torch.tensor(data_df['orc'].values, dtype=torch.float) if 'orc' in data_df.columns else torch.full_like(res_ids, torch.nan, dtype=torch.float),

        'binding_site': torch.tensor(data_df['binding_site'].values, dtype=torch.long) if 'binding_site' in data_df.columns else -torch.ones_like(res_ids, dtype=torch.long),
    }
    return result
