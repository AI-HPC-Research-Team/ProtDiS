import os, sys
import glob
import itertools
import numpy as np
from fastavro import reader
from tqdm import tqdm
import logging
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph, knn_graph

sys.path.append(os.getcwd())
from src.datasets.data_utils import fuse_data

class StructureSimilarity(Dataset):
    def __init__(
        self,
        root: str = './data/tm_align/',
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
        self.protein_index = []
        self.labels = np.load(os.path.join(root, "TMAlignDataset.lddt.npy"))

        avro_path = os.path.join(root, "TMAlignDataset.residue.avro")
        with open(avro_path, "rb") as fo:
            avro_reader = reader(fo)
            for idx, protein_res in tqdm(enumerate(avro_reader), desc="process protein data"):
                protein_info = protein_res['protein'] # dict_keys(['ID', 'sequence', 'EC', 'random_split', 'sequence_split_0.5', 'sequence_split_0.6', 'sequence_split_0.7', 'sequence_split_0.8', 'sequence_split_0.9', 'structure_split_0.3', 'structure_split_0.4', 'structure_split_0.5', 'structure_split_0.6', 'structure_split_0.7', 'structure_split_0.8', 'structure_split_0.9'])

                # apply split filter
                key = 'random_split' if split_type == 'random' else f"{split_type}_split_{redundancy}"
                if protein_info[key] != split:
                    continue

                # load esm_encodings
                pid = protein_info['ID']
                esm_file = os.path.join(root, "esm_encodings", f"{pid}.npz")
                k_file = os.path.join(root, "k_encodings", f"{pid}.npz")
                # add knowledge labels
                tsv_file = os.path.join(root, "labels", f"{pid}.tsv") if use_annos else None

                fused = fuse_data(esm_file, k_file, resolution, tsv_file)
                self.protein_index.append(idx)
                self.data.append((pid, fused))

                if max_samples and len(self.data) >= max_samples: break

        self.protein_pairs = self.compute_pairs(self.protein_index)
        self.idx_pairs = self.compute_pairs(list(range(len(self.data))))

        logging.info(f"Split={split_type}, redundancy={redundancy}, resolution={resolution}: {len(self.data)} proteins in '{split}' set with {self.num_classes} classes, {len(self.protein_pairs)} pairs in all.")

    def __len__(self):
        return len(self.protein_pairs)
    
    @property
    def num_classes(self):
        return 0
    
    def compute_pairs(self, index):
        combinations = np.array(list(itertools.combinations(range(len(index)), 2)), dtype=int)
        return np.array(index)[combinations]

    def __getitem__(self, idx):
        try:
            idx1, idx2 = self.idx_pairs[idx]

            pid1, sample1 = self.data[idx1]
            pid2, sample2 = self.data[idx2]
            y = torch.tensor(self.labels[*self.protein_pairs[idx]])

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
                samples[k] = torch.cat((sample1[k], sample2[k]), dim=0)

            data = Data(
                pid=(pid1, pid2),
                p_idx=torch.cat((torch.zeros(num_nodes1), torch.ones(num_nodes2)), dim=0).int(),  # [num_nodes], protein index
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
        except Exception as e:
            print(f"[ERROR] __getitem__ idx={idx}, pair={self.protein_pairs[idx]} -> {e}")
            raise


# -----------------------------
# Example usage / simple test
# -----------------------------
if __name__ == "__main__":
    # usage example
    dataset = StructureSimilarity(
        split_type='structure',
        redundancy=0.3,
        split='train',
        resolution='residue'
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(dataset[0].pid, dataset[0].y, len(dataset[0].seq))
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

