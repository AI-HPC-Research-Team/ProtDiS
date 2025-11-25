import os, sys, glob
import logging
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

sys.path.append(os.getcwd())
from src.datasets.data_utils import SS_MAP, merge_data
from src.datasets.filter_metadata import filter_samples

class PretrainDataset(Dataset):
    def __init__(self, pdb_roots: list, pdb_num:int = None, max_nodes:int = None, read_kf=False, lazy = True):
        self.pdb_num = pdb_num or float('inf') # limit number of PDB entries
        self.max_nodes = max_nodes or float('inf') # set to 1000 to avoid OOM
        self.read_kf = read_kf # load knowledge embeddings
        self.lazy = lazy # enable lazy loading
        self.features = ['ss', 'asa', 'bfactor', 'lp', 'hydro', 'stability', 'entropy', 'orc']

        # gather all train samples
        self.samples = []
        for root in pdb_roots:
            list_path = os.path.join(root, 'filtered_samples.tsv')
            if not os.path.exists(list_path):
                filter_samples(root, 'pred' if 'alphafolddb' in root else 'exp', output_file='filtered_samples.tsv')

            with open(list_path, 'r') as f:
                lines = f.readlines()[1:]
            if len(lines) > self.pdb_num:
                lines = random.sample(lines, k=self.pdb_num)
            samples = [tuple(line.rstrip('\n').split('\t')) for line in lines]
            self.samples.extend(samples)
        logging.info(f"Found {len(self.samples)} valid samples.")

        if not self.lazy:
            self.data = []
            all_residues = []
            
            # 流式处理避免内存爆炸
            for esm_p, label_p in tqdm(self.samples, desc="Collecting residues"):
                merged = merge_data(esm_p, tsv_file=label_p, k_file=esm_p.replace('esm_', 'k_') if self.read_kf else None)
                n = len(merged)
                
                # 添加蛋白质ID到每个残基
                for i in range(n):
                    all_residues.append(
                        merged.iloc[i]
                    )

            logging.info(f"Successfully loaded {len(self.samples)} samples into memory, {len(all_residues)} residues in total.")
            
            # 随机打乱残基
            random.shuffle(all_residues)
            
            # 分批创建样本
            for i in tqdm(range(0, len(all_residues), 150), desc="Creating samples"):
                residues = all_residues[i:i+150]
                
                # 重组为样本
                merged_rows = residues
                merged_df = pd.DataFrame(merged_rows)
                
                # 使用第一个蛋白质ID作为标识（或创建新ID）
                sample_id = f"mixed_batch_{i//150}"
                
                #self.data.append((sample_id, merged_df))
                self.data.append(self.collect_samples(sample_id, merged_df))

            logging.info(f"Successfully divided {len(all_residues)} residues into {len(self.data)} samples.")

        self.collator = Collater(self)
    
    def __len__(self):
        return len(self.samples) if self.lazy else len(self.data)
    
    def __getitem__(self, idx):
        if self.lazy:
            esm_p, label_p = self.samples[idx]
            file_name = os.path.basename(esm_p)[:-4]
            merged = merge_data(esm_p, tsv_file=label_p, k_file=esm_p.replace('esm_', 'k_') if self.read_kf else None)
            data = self.collect_samples(file_name, merged)
        else:
            data = self.data[idx]
        return data
    
    def collect_samples(self, file_name, merged):
        s_feat = torch.tensor(np.vstack(merged['structure_encodings'].values), dtype=torch.float, device='cpu')
        #s_feat = s_feat + torch.randn_like(s_feat) * 0.05
        coords = torch.tensor(np.vstack(merged['coordinates'].values), dtype=torch.float, device='cpu')

        labels = {}
        # map ss to int
        labels['ss_label'] = torch.tensor([SS_MAP.get(x, -1) for x in merged['ss']], dtype=torch.long, device='cpu')
        # float features
        labels['asa_label'] = torch.tensor(merged['asa'].values, dtype=torch.float, device='cpu')
        labels['bfactor_label'] = torch.tensor(merged['bfactor'].values, dtype=torch.float, device='cpu')
        labels['lp_label'] = torch.tensor(merged['local_packing'].values, dtype=torch.float, device='cpu')
        labels['hydro_label'] = torch.tensor(merged['hydrophobicity'].values, dtype=torch.float, device='cpu')
        labels['stability_label'] = torch.tensor(merged['stability'].values, dtype=torch.float, device='cpu')
        labels['entropy_label'] = torch.tensor(merged['contact_entropy'].values, dtype=torch.float, device='cpu')
        labels['orc_label'] = torch.tensor(merged['orc'].values, dtype=torch.float, device='cpu')

        if self.read_kf:
            ck_encodings = torch.tensor(np.vstack(merged['ck_encodings'].values), dtype=torch.float, device='cpu')
            ss_encodings = torch.tensor(np.vstack(merged['ss_encodings'].values), dtype=torch.float, device='cpu')
            asa_encodings = torch.tensor(np.vstack(merged['asa_encodings'].values), dtype=torch.float, device='cpu')
            bfactor_encodings = torch.tensor(np.vstack(merged['bfactor_encodings'].values), dtype=torch.float, device='cpu')
            lp_encodings = torch.tensor(np.vstack(merged['lp_encodings'].values), dtype=torch.float, device='cpu')
            hydro_encodings = torch.tensor(np.vstack(merged['hydro_encodings'].values), dtype=torch.float, device='cpu')
            stability_encodings = torch.tensor(np.vstack(merged['stability_encodings'].values), dtype=torch.float, device='cpu')
            entropy_encodings = torch.tensor(np.vstack(merged['entropy_encodings'].values), dtype=torch.float, device='cpu')
            data = Data(
                file=file_name,
                s_enc=s_feat,         # structure encodings
                ck_enc = ck_encodings,  # common knowledge encodings
                ss_enc = ss_encodings,  # secondary structure encodings
                asa_enc = asa_encodings,  # ASA encodings
                bfactor_enc = bfactor_encodings,  # B-factor encodings
                lp_enc = lp_encodings,  # local packing encodings
                hydro_enc = hydro_encodings,  # hydrophobicity encodings
                stability_enc = stability_encodings,  # stability encodings
                entropy_enc = entropy_encodings,  # contact entropy encodings
                pos=coords,         # 3D coordinates
                **labels
            )
        else:
            data = Data(
                file=file_name,
                s_enc=s_feat,         # structure encodings
                pos=coords,         # 3D coordinates
                **labels
            )
        return data
    
    def collate_fn(self, batch):
        data = self.collator(batch)
        # 数据采样
        n_nodes = len(data.batch)
        if n_nodes > self.max_nodes:
            # 随机采样1000个节点
            indices = torch.randperm(n_nodes)[:self.max_nodes]
            for key in ['s_enc', 'pos', 'batch'] + [f"{feat}_label" for feat in self.features]:
                setattr(data, key, getattr(data, key)[indices])
        return data
    

# ------------------------ Test script ------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Test PretrainDataset and DataLoader")
    parser.add_argument('--pdb_root', type=str, default="data/CATHv44-S40/", help='Path to CATH PDB ESM root')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # instantiate dataset and loader
    dataset = PretrainDataset([args.pdb_root])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn) #, num_workers=4

    # iterate one batch
    for data in loader:
        print(getattr(data, 'file'))
        print(f"Batch graphs: {data.num_graphs}")
        print("Graph node count:", data.batch.bincount())
        for key in ['s_enc', 'pos', 'batch'] + [f"{feat}_label" for feat in dataset.features]:
            tensor = getattr(data, key)
            print(f"{key}: {tensor.shape} - {tensor.dtype} - {tensor.min()} - {tensor.max()}")
        print("Example ss labels (first 10):", data.ss_label[:10])
        print("Example batch id (first 10):", data.batch[:10])
        #print("Example coordinates (first 10):", batch.pos[:10])
        #print("Example structure encodings (first node):", batch.s_enc[0])
        break

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    for data in loader:
        for key in ['stability_label']: # no NaN for stability
            tensor = getattr(data, key)
            if torch.isnan(tensor).any():
                print(f"Warning: {key} contains NaN values! in {data.file}")
                exit(1)

        # inspect first graph
        if len(getattr(data, 's_enc')) != len(getattr(data, 'ss_label')): break
