import os, sys
import glob
import itertools
import numpy as np
import json
from fastavro import reader
from tqdm import tqdm
import logging
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph, knn_graph

sys.path.append(os.getcwd())
from src.datasets.data_utils import fuse_data

class ProteinProteinInterface(Dataset):
    def __init__(
        self,
        root: str = './data/ppis/',
        split_type: str = 'random',
        redundancy: float = 0.7,
        split: str = 'train',
        resolution: str = 'residue',
        use_annos: bool = False,
        max_samples: int = None,
        graph: str = 'eps',
        radius: float = 8.0,
    ):
        assert split in ['train', 'test', 'val'], f"Invalid split: {split}. Must be one of ['train', 'test', 'val']"
        assert split_type in ['random', 'sequence', 'structure'], f"Invalid split_type: {split_type}. Must be one of ['random', 'sequence', 'structure']"
        if split_type == 'sequence':
            assert redundancy in [0.5, 0.6, 0.7, 0.8, 0.9], f"Invalid redundancy: {redundancy}. Must be one of [0.5, 0.6, 0.7, 0.8, 0.9]"
        elif split_type == 'structure':
            assert redundancy in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], f"Invalid redundancy: {redundancy}. Must be one of [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]"
        assert resolution in ['residue', 'atom'], f"Invalid resolution: {resolution}. Must be one of ['residue', 'atom']"

        self.split_type = split_type
        self.redundancy = redundancy
        self.split = split
        self.resolution = resolution
        self.graph = graph
        self.radius = radius

        self.data = []
        with open(os.path.join(root, "interfaces.json"), 'r') as f:
            self.interfaces = json.load(f)

        avro_path = os.path.join(root, "ProteinProteinInterfaceDataset.residue.avro")
        with open(avro_path, "rb") as fo:
            avro_reader = reader(fo)
            for idx, protein_res in tqdm(enumerate(avro_reader), desc="process protein data"):
                protein_info = protein_res['protein'] # dict_keys(['ID', 'sequence', 'EC', 'random_split', 'sequence_split_0.5', 'sequence_split_0.6', 'sequence_split_0.7', 'sequence_split_0.8', 'sequence_split_0.9', 'structure_split_0.3', 'structure_split_0.4', 'structure_split_0.5', 'structure_split_0.6', 'structure_split_0.7', 'structure_split_0.8', 'structure_split_0.9'])
                residue_info = protein_res['residue']

                # apply split filter
                key = 'random_split' if split_type == 'random' else f"{split_type}_split_{redundancy}"
                if protein_info[key] != split:
                    continue

                # load esm_encodings
                pid = protein_info['ID']
                esm_file = os.path.join(root, "esm_encodings", f"{pid.split('_')[0]}{pid.split('_')[1]}.npz")
                k_file = os.path.join(root, "k_encodings", f"{pid.split('_')[0]}{pid.split('_')[1]}.npz")
                # add knowledge labels
                tsv_file = os.path.join(root, "labels", f"{pid.split('_')[0]}{pid.split('_')[1]}.tsv") if use_annos else None

                if not os.path.exists(esm_file):
                    print(f"Warning: esm_encodings file {esm_file} does not exist, skipping {pid}")
                    continue
                fused = fuse_data(esm_file, k_file, resolution, tsv_file, None, residue_info['chain_id'][0])
                if fused is None:
                    print(f"Warning: no data for {pid}, skipping")
                    continue
                self.data.append((pid, fused))

                if max_samples and len(self.data) >= max_samples: break

        self.chain_pairs = self.compute_pairs()
        logging.info(f"Split={split_type}, redundancy={redundancy}, resolution={resolution}: {len(self.data)} protein chains in '{split}' set with {self.num_classes} classes, {len(self.chain_pairs)} pairs in all.")

    def __len__(self):
        return len(self.chain_pairs)
    
    @property
    def num_classes(self):
        return 0
    
    def compute_pairs(self):
        """ Grab all pairs of chains that share an interface"""
        protein_to_index = {s[0]: i for i, s in enumerate(self.data)}
        def find_index(pdbid, chain):
            return protein_to_index[f'{pdbid}_{chain}']

        chain_pairs = []
        for i, (pid, fused) in enumerate(self.data):
            pdbid, chain = pid.split('_')
            try:
                chain_pairs.extend([(i, find_index(pdbid, partner)) for partner in self.interfaces[pdbid][chain]])
            # if chain is not in any interface, we skip
            except (KeyError, IndexError):
                continue
        return np.array(chain_pairs, dtype=int)

    def __getitem__(self, idx):
        idx1, idx2 = self.chain_pairs[idx]

        pid1, sample1 = self.data[idx1]
        pid2, sample2 = self.data[idx2]

        # create contact matrix
        pdbid, chain1 = pid1.split('_')
        _, chain2 = pid2.split('_')
        inds = torch.tensor(self.interfaces[pdbid][chain1][chain2])

        min_index1 = min(sample1['res_id'].min(), inds[:, 0].min())
        res_id1 = sample1['res_id'] - min_index1
        inds[:, 0] = inds[:, 0] - min_index1
        max_index1 = max(res_id1.max(), inds[:, 0].max())

        min_index2 = min(sample2['res_id'].min(), inds[:, 1].min())
        res_id2 = sample2['res_id'] - min_index2
        inds[:, 1] = inds[:, 1] - min_index2
        max_index2 = max(res_id2.max(), inds[:, 1].max())

        contacts = torch.zeros((max_index1 + 1, max_index2 + 1))
        contacts[inds[:, 0], inds[:, 1]] = 1
        contacts = contacts[res_id1, :][:, res_id2]  # [num_nodes1, num_nodes2]
        flat_contacts = contacts.flatten()  # [num_nodes1 * num_nodes2]

        if self.split == 'train':
            y = self.create_balanced_sample(flat_contacts)
        else:
            y = flat_contacts

        num_nodes1 = len(sample1['sequence'])
        num_nodes2 = len(sample2['sequence'])

        pos1 = sample1['coordinates']
        pos2 = sample2['coordinates']
        # optional data augmentation for training
        if self.split == 'train':
            pos1 = pos1 + torch.randn_like(pos1) * 0.05
            pos2 = pos2 + torch.randn_like(pos2) * 0.05

        if self.resolution == 'residue':
            if self.graph == 'eps':
                edge_index = torch.cat((radius_graph(pos1, r=self.radius, loop=False), 
                                        radius_graph(pos2, r=self.radius, loop=False)), dim=1)
            elif self.graph == 'knn':
                k = 7
                edge_index = torch.cat((knn_graph(pos1, k=min(len(pos1) - 1, k), loop=False),
                                        knn_graph(pos2, k=min(len(pos2) - 1, k), loop=False)), dim=1)
        else:
            edge_index = None

        samples = {}
        for k in sample1.keys():
            if k =='chain_id':
                samples[k] = np.concatenate((sample1[k], sample2[k]), axis=0)
            else:
                samples[k] = torch.cat((sample1[k], sample2[k]), dim=0)

        data = Data(
            pid=(pid1, pid2),
            p_idx=torch.cat((torch.zeros(num_nodes1), torch.ones(num_nodes2)), dim=0).int(),  # [num_nodes], protein index
            chain_id=samples['chain_id'],
            res_id=samples['res_id'],
            # basis features
            seq=samples['sequence'],
            s=samples['structure_encodings'],
            # knowledge features
            ck=samples['ck_encodings'],
            ss=samples['ss_encodings'],
            asa=samples['asa_encodings'],
            bfactor=samples['bfactor_encodings'],
            lp=samples['lp_encodings'],
            hydro=samples['hydro_encodings'],
            stability=samples['stability_encodings'],
            entropy=samples['entropy_encodings'],
            orc=samples['orc_encodings'],
            # labels input
            ss_label=samples['ss_label'],
            asa_label=samples['asa_label'],
            bfactor_label=samples['bfactor_label'],
            lp_label=samples['lp_label'],
            hydro_label=samples['hydro_label'],
            stability_label=samples['stability_label'],
            entropy_label=samples['entropy_label'],
            orc_label=samples['orc_label'],
            # graph data
            aid=torch.cat([torch.arange(num_nodes1), torch.arange(num_nodes2)], dim=0).unsqueeze(1).float(),  # [num_nodes, 1], node index
            ori=samples['ori'],  # [num_nodes, 3, 3]

            pos=torch.cat((pos1, pos2), dim=0),
            edge_index=edge_index,
            edge_attr=None,
            y=y,
        )
        return data
    
    @staticmethod
    def create_balanced_sample(flat_contacts):
        pos_indices = torch.where(flat_contacts > 0)[0]
        num_pos = len(pos_indices)

        # 获取负样本索引
        neg_indices = torch.where(flat_contacts == 0)[0]
        
        # 采样与正样本相同数量的负样本
        if len(neg_indices) > num_pos:
            selected_neg = torch.randperm(len(neg_indices))[:num_pos]
            sampled_neg_indices = neg_indices[selected_neg]
        else:
            sampled_neg_indices = neg_indices
        
        # 创建平衡样本索引
        balanced_indices = torch.cat([pos_indices, sampled_neg_indices])

        # create a mask for the balanced indices
        #mask = torch.zeros_like(flat_contacts, dtype=torch.bool)
        #mask[balanced_indices] = True
        
        # 创建平衡标签（只包含选中样本，其他设为-1表示忽略）
        y = torch.full_like(flat_contacts, -1.0)
        y[balanced_indices] = flat_contacts[balanced_indices]
        return y


# -----------------------------
# Example usage / simple test
# -----------------------------
if __name__ == "__main__":
    # usage example
    dataset = ProteinProteinInterface(
        split_type='structure',
        redundancy=0.3,
        split='train',
        resolution='residue'
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(dataset[0].pid, dataset[0].y, len(dataset[0].seq))
    print(torch.where(dataset[0].y > 0))
    print(dataset[0].p_idx)
    for data in data_loader:
        assert len(data.seq) == len(data.pos)
        assert len(data.seq) == data.num_nodes
        assert len(data.seq) == len(data.s)
        assert len(data.seq) == len(data.ck)
        assert len(data.seq) == len(data.ss)
        assert len(data.seq) == len(data.asa)
        assert len(data.seq) == len(data.bfactor)
        assert len(data.seq) == len(data.hbond)
        assert len(data.seq) == len(data.contact)
        assert len(data.seq) == len(data.hydro)
        assert len(data.seq) == len(data.ss_label)
        assert len(data.seq) == len(data.asa_label)
        assert len(data.seq) == len(data.bfactor_label)
        assert len(data.seq) == len(data.hbond_label)
        assert len(data.seq) == len(data.contact_label)
        assert len(data.seq) == len(data.hydro_label)

        print(data)
        print(data.num_graphs)
        print(data.seq)
        print(data.y)
        print(data.pos)
        print(data.s)
        print(data.ck)
        print(data.ss)
        print(data.asa)
        print(data.bfactor)
        print(data.hbond)
        print(data.contact)
        print(data.hydro)
        print(data.ss_label)
        print(data.asa_label)
        print(data.bfactor_label)
        print(data.hbond_label)
        print(data.contact_label)
        print(data.hydro_label)
        break

