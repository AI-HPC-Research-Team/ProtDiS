import os
import numpy as np
import pandas as pd
import torch

SSE_8CLASS_VOCAB = "GHITEBSC"
SSE_3CLASS_VOCAB = "HEC"
SSE_8CLASS_TO_3CLASS_MAP = {
    "G": "H",
    "H": "H",
    "I": "H",
    "T": "C",
    "E": "E",
    "B": "E",
    "S": "C",
    "C": "C",
}

SS_LIST = ['H: alpha helix', 'B: beta bridge', 'E: extended strand', 'G: 3_10 helix', 'I: π-helix', 'T: turn', 'S: bend', 'C: coil']
SS_LETTERS = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']
SS_MAP = {c:i for i,c in enumerate(SS_LETTERS)}
SS_MAP['-'] = SS_MAP['C']

ASA_DISCRETIZATION_BOUNDARIES = [0, 0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4, 336.]
BFACTOR_DISCRETIZATION_BOUNDARIES = [-20, 8.6, 11.7, 14.1, 16.1, 18.0, 19.8, 21.6, 23.4, 25.3, 27.2, 29.2, 31.3, 33.5, 35.9, 38.4, 41.2, 44.2, 47.5, 51.4, 55.9, 61.4, 68.8, 80.6, 150., 570.]
#LP_DISCRETIZATION_BOUNDARIES = [3.76, 6.0, 6.12, 6.21, 6.29, 6.36, 6.44, 6.51, 6.59, 6.68, 6.76, 6.85, 6.94, 7.03, 7.12, 7.21, 7.31, 7.42, 7.54, 7.69, 7.87, 8.1, 8.43, 9.05, 16.89]
LP_DISCRETIZATION_BOUNDARIES = [0.035, 0.399, 0.458, 0.5, 0.534, 0.564, 0.59, 0.615, 0.638, 0.661, 0.684, 0.706, 0.729, 0.752, 0.776, 0.8, 0.826, 0.854, 0.884, 0.918, 0.958, 1.006, 1.068, 1.158, 9.455]
HYDRO_DISCRETIZATION_BOUNDARIES = [-4.5, -2.12, -1.71, -1.44, -1.23, -1.05, -0.89, -0.74, -0.6, -0.47, -0.34, -0.22, -0.1, 0.02, 0.14, 0.27, 0.4, 0.54, 0.69, 0.85, 1.02, 1.24, 1.5, 1.88, 4.5]
STABILITY_DISCRETIZATION_BOUNDARIES = [0., 0.92, 1.08, 1.17, 1.23, 1.28, 1.32, 1.35, 1.38, 1.41, 1.44, 1.465, 1.49, 1.51, 1.53, 1.55, 1.575, 1.6, 1.62, 1.645, 1.67, 1.7, 1.74, 1.8, 5.13]
ENTROPY_DISCRETIZATION_BOUNDARIES = [0., 0.485, 0.544, 0.565, 0.590, 0.613, 0.637, 0.655, 0.667, 0.711, 0.721, 0.763, 0.931]
ORC_DISCRETIZATION_BOUNDARIES = [-0.379, -0.069, -0.049, -0.036, -0.025, -0.015, -0.005, 0.004, 0.013, 0.022, 0.031, 0.041, 0.051, 0.061, 0.072, 0.084, 0.096, 0.11, 0.124, 0.141, 0.159, 0.183, 0.214, 0.259, 1.]

# calculate the weights of labels
ss_freq = np.array([0.3180, 0.0114, 0.2181, 0.0381, 0.0059, 0.1128, 0.0886, 0.2071])
median_ss_freq = (0.0886+0.1128) / 2
ss_weights = median_ss_freq / ss_freq
ss_weights = ss_weights / ss_weights.max()

# calculate the median and scale for each feature
asa_mean, asa_std = 55.99, 50.07
bfactor_mean, bfactor_std = 29.99, 20.15
lp_mean, lp_std = 0.74, 0.22
hydro_mean, hydro_std = -0.11, 1.15
stability_mean, stability_std = 1.44, 0.28
entropy_mean, entropy_std = 0.62, 0.13
orc_mean, orc_std = 0.07, 0.10


def merge_data(esm_file, k_file=None, tsv_file=None, resolution='atom', require_label=True, chain_id=None):
    """
    Merge esm encoding and label data into a single Data object.
    """
    # load esm encodings
    if esm_file.endswith('.npz'):
        esm_data = np.load(esm_file, allow_pickle=True)
    else:
        esm_data = torch.load(esm_file, map_location='cpu')
        esm_data['structure_encodings'] = esm_data['structure_encodings'].numpy()
    pd_data = {
        'chain_id': esm_data['chain_id'].tolist(),
        'res_id': esm_data['residue_index'].tolist(),
        'ins_code': esm_data['insertion_code'].tolist(),
        'res': esm_data['residue_type'].tolist(),
        'coordinates': list(esm_data['coordinates']),
        'structure_encodings': list(esm_data['structure_encodings']),
    }

    if k_file is not None:
        k_encodings = torch.load(k_file, map_location='cpu')

        pd_data.update(
            {f'{fea}_encodings': list(k_encodings[f'{fea}_encodings']) for fea in k_encodings['feature_names']}
            )

    # build esm DataFrame for merge
    merged = pd.DataFrame(pd_data)
    if chain_id is not None:
        merged = merged[merged['chain_id'] == chain_id]
        if len(merged) == 0:
            return None  # no data for this chain_id

    # read labels
    if tsv_file is not None:
        label_df = pd.read_csv(tsv_file, sep='\t')
        label_df = label_df.rename(columns={
            'Chain': 'chain_id',
            'Residue ID': 'res_id',
            'Insertion Code': 'ins_code',
            'Residue': 'res',
        })
        label_df['chain_id'] = label_df['chain_id'].apply(lambda col: str(col).strip())
        label_df['ins_code'] = label_df['ins_code'].apply(lambda x: x.replace('-', ''))
        
        # 合并标签数据
        merged = merged.merge(
            label_df,
            on=['chain_id', 'res_id', 'ins_code', 'res'], 
            how='inner' if require_label else 'left', 
            sort=False, 
        )

    merged = merged.drop_duplicates(subset=['chain_id', 'res_id', 'ins_code', 'res'], keep='first')  # remove duplicates

    # process different resolutions
    if resolution == 'residue':
        ca_coords = np.array([pos[1] for pos in merged['coordinates']])
        valid_mask = ~np.isnan(ca_coords).any(axis=1) # request Ca coordinates
        
        merged = merged[valid_mask]
        ca_coords = ca_coords[valid_mask]
        merged['coordinates'] = list(ca_coords)

        return merged
    
    elif resolution == 'atom':
        return merged
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

