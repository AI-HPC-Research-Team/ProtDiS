import os, sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
from src.datasets.data_utils import SS_MAP

class CathDataset(Dataset):
    def __init__(self, root:str):
        esm_data_root = os.path.join(root, "esm_encodings/dompdb-S40")
        superfamily_anno_path = os.path.join(root, "more_labels/dompdb-S40-superfamily.pkl")

        # 读取预处理的domain标注
        with open(superfamily_anno_path, "rb") as f:
            cath_domain_annos = pickle.load(f)

        unit_sequences = []
        coordinates = []
        ss_labels = []
        sasa_labels = []
        b_factor_labels = []
        s_encodings = []
        superfamily_labels = []
        cnt = 0
        for file_name in tqdm(os.listdir(esm_data_root), desc="load CATH dataset"):
            filepath = os.path.join(esm_data_root, file_name)
            esm_data = torch.load(filepath, map_location='cpu') # sequence_tokens和structure_tokens的第一个、最后一个token是special token
            unit_sequences.append(esm_data['structure_tokens'][1:-1])
            coordinates.append(esm_data['coordinates'])
            ss_labels.append(np.array([SS_MAP[ss] for ss in esm_data['secondary_structure']]))
            sasa_labels.append(np.array(esm_data['sasa'], dtype=np.float32))
            b_factor_labels.append(esm_data['b_factor'])
            s_encodings.append(esm_data['structure_encodings'].squeeze(0))

            domain = file_name[:7]
            if domain in cath_domain_annos:
                superfamily_labels.append(cath_domain_annos[domain]['superfamily_index'])
                cnt += 1
            else:
                superfamily_labels.append(-1)
        print(f"Load CATH data finished, {len(unit_sequences)} items in all, with {cnt} valid superfamily labels.")

        self.unit_sequences = unit_sequences
        self.coordinates = coordinates
        self.ss_labels = ss_labels
        self.sasa_labels = sasa_labels
        self.b_factor_labels = b_factor_labels
        self.s_encodings = s_encodings
        self.superfamily_labels = superfamily_labels
    
    def __len__(self):
        return len(self.unit_sequences)
    
    def __getitem__(self, idx):
        unit_sequence = self.unit_sequences[idx] # (l,)
        coordinates = self.coordinates[idx] # (l, 37, 3)
        ss_label = self.ss_labels[idx] # (l,)
        sasa_label = self.sasa_labels[idx] # (l,)
        b_factor_label = self.b_factor_labels[idx] # (l,)
        s_encodings = self.s_encodings[idx] # (l, 128)
        superfamily_label = self.superfamily_labels[idx] # (,)
        return {
            'unit_sequence': unit_sequence,
            'coordinates': coordinates,
            'ss_label': ss_label,
            'sasa_label': sasa_label,
            'b_factor_label': b_factor_label,
            's_encodings': s_encodings,
            'superfamily_label': superfamily_label,
        }
    
    @staticmethod
    def collate_fn(batch):
        unit_sequence = []
        coordinates = []
        ss_label = []
        sasa_label = []
        b_factor_label = []
        s_encodings = []
        superfamily_label = []
        sequence_id = []

        # 拼接每个样本的序列数据
        for current_seq_id, data in enumerate(batch):
            unit_sequence.append(data['unit_sequence'])
            coordinates.append(data['coordinates'])
            ss_label.append(data['ss_label'])
            sasa_label.append(data['sasa_label'])
            b_factor_label.append(data['b_factor_label'])
            s_encodings.append(data['s_encodings'])
            superfamily_label.append(data['superfamily_label'])
            # 为每个 token 分配一个 sequence_id
            sequence_id.extend([current_seq_id] * len(data['unit_sequence']))  # 对每个token使用相同的seq_id

        # 使用 numpy 拼接数据
        unit_sequence = np.concatenate(unit_sequence)  # 拼接成一个长向量 (total_length,)
        coordinates = np.concatenate(coordinates)        # 拼接成 (total_length, 37, 3)
        ss_label = np.concatenate(ss_label)            # (total_length,)
        sasa_label = np.concatenate(sasa_label)        # (total_length,)
        b_factor_label = np.concatenate(b_factor_label) # (total_length,)
        s_encodings = np.concatenate(s_encodings)         # (total_length, 128)
        superfamily_label = np.array(superfamily_label)    # (batch_size,)
        sequence_id = np.array(sequence_id)            # (total_length,)

        return {
            'unit_sequence': torch.tensor(unit_sequence),      # (total_length,)
            'coordinates': torch.tensor(coordinates),             # (total_length, 37, 3)
            'ss_label': torch.tensor(ss_label),                 # (total_length,)
            'sasa_label': torch.tensor(sasa_label),             # (total_length,)
            'b_factor_label': torch.tensor(b_factor_label),     # (total_length,)
            's_encodings': torch.tensor(s_encodings),             # (total_length, 128)
            'superfamily_label': torch.tensor(superfamily_label), # (batch_size,)
            'sequence_id': torch.tensor(sequence_id),           # (total_length,)
        }

        

# -----------------------------
# 以下为简单的测试示例
# -----------------------------
if __name__ == "__main__":
    # 创建数据集和数据加载器
    dataset = CathDataset("data/CATHv44/")
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=True)

    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx + 1} - Number of sequences: {len(batch['superfamily_label'])} - Total length: {len(batch['unit_sequence'])}")

        for key, value in batch.items():
            print(f"{key}: {value.shape} - {value.dtype}")
        
        print(f"Example Unit Sequence: {batch['unit_sequence']}")
        print(f"Example Coordinates: {batch['coordinates'][0]}")
        print(f"Example Superfamily: {batch['superfamily_label']}")
        
        break
