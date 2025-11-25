import os, sys
import glob
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

class ProteinFamily(Dataset):
    def __init__(
        self,
        root: str = './data/protein_family/',
        split_type: str = 'random',
        redundancy: float = 0.7,
        split: str = 'train',
        resolution: str = 'residue',
        use_annos: bool = False,
        max_samples: int = None,
        graph: str = 'eps',
        radius: float = 8.0,
        lazy = False,
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
        self.lazy = lazy

        self.data = []
        raw_labels = []
        label_set = set()

        avro_path = os.path.join(root, "ProteinFamilyDataset.residue.avro")
        with open(avro_path, "rb") as fo:
            avro_reader = reader(fo)
            for protein_res in tqdm(avro_reader, desc="process protein data"):
                protein_info = protein_res['protein'] # dict_keys(['ID', 'sequence', 'EC', 'random_split', 'sequence_split_0.5', 'sequence_split_0.6', 'sequence_split_0.7', 'sequence_split_0.8', 'sequence_split_0.9', 'structure_split_0.3', 'structure_split_0.4', 'structure_split_0.5', 'structure_split_0.6', 'structure_split_0.7', 'structure_split_0.8', 'structure_split_0.9'])
                label_set.add(protein_info['Pfam'][0])

                # apply split filter
                key = 'random_split' if split_type == 'random' else f"{split_type}_split_{redundancy}"
                if protein_info[key] != split:
                    continue

                # load esm_encodings
                pid = protein_info['ID']
                esm_files = glob.glob(os.path.join(root, "esm_encodings", f"{pid}*.pth"))
                if len(esm_files)!=1:
                    print(f"Warning: found {len(esm_files)} esm_encodings for {pid}, skipping")
                    continue
                tsv_file = os.path.join(root, "labels", f"{pid}.tsv") if use_annos else None

                if lazy:
                    self.data.append((pid, (esm_files[0], esm_files[0].replace("esm_encodings", "k_encodings").replace('.pth', '.npz'), resolution, tsv_file)))
                else:
                    fused = fuse_data(esm_files[0], esm_files[0].replace("esm_encodings", "k_encodings").replace('.pth', '.npz'), resolution, tsv_file)
                    self.data.append((pid, fused))
                raw_labels.append(protein_info['Pfam'][0])

                if max_samples and len(self.data) >= max_samples: break

        # map string labels to integers
        self.token_map = {label: i for i, label in enumerate(sorted(label_set))}
        self.labels = [self.token_map[label]for label in raw_labels]
        logging.info(f"Split={split_type}, redundancy={redundancy}, resolution={resolution}: {len(self.data)} proteins in '{split}' set with {self.num_classes} classes.")

    def __len__(self):
        return len(self.data)
    
    @property
    def num_classes(self):
        return len(self.token_map)

    def __getitem__(self, idx):
        if self.lazy:
            pid, (esm_p, k_p, resolution, tsv_file) = self.data[idx]
            sample = fuse_data(esm_p, k_p, resolution, tsv_file)
        else:
            pid, sample = self.data[idx]
        num_nodes = len(sample['sequence'])

        # build node features and positions
        pos = sample['coordinates']
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # optional data augmentation for training
        if self.split == 'train':
            pos = pos + torch.randn_like(pos) * 0.05

        if self.resolution == 'residue':
            if self.graph == 'eps':
                edge_index = radius_graph(pos, r=self.radius, loop=False)
            elif self.graph == 'knn':
                k = 7
                num_neighbors = min(len(pos) - 1, k)
                edge_index = knn_graph(pos, k=num_neighbors, loop=False)
        else:
            edge_index = None

        data = Data(
            pid=pid,
            # basis features
            seq=sample['sequence'],
            s=sample['structure_encodings'],
            # knowledge features
            ck=sample['ck_encodings'],
            ss=sample['ss_encodings'],
            asa=sample['asa_encodings'],
            bfactor=sample['bfactor_encodings'],
            lp=sample['lp_encodings'],
            hydro=sample['hydro_encodings'],
            stability=sample['stability_encodings'],
            entropy=sample['entropy_encodings'],
            orc=sample['orc_encodings'],
            # labels input
            ss_label=sample['ss_label'],
            asa_label=sample['asa_label'],
            bfactor_label=sample['bfactor_label'],
            lp_label=sample['lp_label'],
            hydro_label=sample['hydro_label'],
            stability_label=sample['stability_label'],
            entropy_label=sample['entropy_label'],
            orc_label=sample['orc_label'],
            # graph data
            aid=torch.arange(num_nodes).unsqueeze(1).float(),  # [num_nodes, 1], node index
            ori=sample['ori'],  # [num_nodes, 3, 3]
            pos=pos,
            edge_index=edge_index,
            edge_attr=None,
            y=y,
        )
        return data


# -----------------------------
# Example usage / simple test
# -----------------------------
if __name__ == "__main__":
    # usage example
    dataset = ProteinFamily(
        root="./data/protein_family/",
        split_type='structure',
        redundancy=0.3,
        split='val',
        resolution='residue'
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(dataset[0].pid, dataset[0].y, len(dataset[0].seq))
    for data in data_loader:
        assert len(data.seq) == len(data.pos)
        assert len(data.seq) == data.num_nodes
        assert len(data.seq) == len(data.s)
        assert len(data.seq) == len(data.ck)
        assert len(data.seq) == len(data.ss)
        assert len(data.seq) == len(data.asa)
        assert len(data.seq) == len(data.bfactor)
        assert len(data.seq) == len(data.ss_label)
        assert len(data.seq) == len(data.asa_label)
        assert len(data.seq) == len(data.bfactor_label)
        assert len(data.seq) == len(data.hbond_label)
        assert len(data.seq) == len(data.contact_label)
        assert len(data.seq) == len(data.hydro_label)

        print(data)
        print(data.seq)
        print(data.y)
        print(data.pos)
        print(data.s)
        print(data.ck)
        print(data.ss)
        print(data.asa)
        print(data.bfactor)
        print(data.ss_label)
        print(data.asa_label)
        print(data.bfactor_label)
        print(data.hbond_label)
        print(data.contact_label)
        print(data.hydro_label)
        break

