# ProtDiS: Protein Disentangled Structure Representation

**ProtDiS** is a framework for learning *knowledge-guided, disentangled structural representations* of protein micro-environments.  
Instead of embedding protein structures into opaque high-dimensional vectors, ProtDiS decomposes each local environment into **biophysically meaningful channels**—such as solvent exposure, packing density, curvature, flexibility, and stability—forming a human-understandable latent space that can be inspected, edited, and applied across tasks.

---

## 🧠 Overview

ProtDiS enables a new form of **knowledge-aligned structural representation learning**, supporting:

### ✔ Knowledge-aligned structural analysis  
Decomposed knowledge channels (e.g., solvent exposure, packing density, curvature, flexibility, stability) allow structural variation to be examined along biological axes rather than black-box latent components.

### ✔ Fine-grained functional discrimination among similar folds  
Even when two proteins share nearly identical global folds (high TM-score), ProtDiS reveals micro-environment differences that correspond to functional divergence (e.g., enzyme subclasses, binding preferences).

### ✔ Improved performance on downstream structure–function tasks  
ProtDiS selectively fuses **task-relevant knowledge channels**, improving generalization, especially under structure-based data splits.

### ✔ (In Progress) Interpretable feature manipulation for modeling and design  
Biophysical dimensions will serve as directions for controlled edits to protein latent space.

---

## ⚙️ Environment Setup

We recommend using a virtual environment or Conda.

```bash
conda create -n protein_env python=3.9
conda activate protein_env
```

Dependencies include:

- `numpy`, `pandas`, `scikit-learn`
- `biopython`
- `torch`
- `tqdm`, `matplotlib`, `seaborn`


## 📦 Data Preparation

The model takes protein structural data as input (e.g., PDB/CIF files). Each protein is decomposed into local knowledge embeddings.

### Step 1: Download Protein Structures

You can download original structures from the PDB/AlphafoldDB or use datasets like SCOPe or CATH.

### Step 2: Preprocess Structures

The data preprocessing scripts will be made available in the future.


## 🏋️‍♂️ Training

To train the Knowledge Orthogonal Network:

```bash
cd ProtDiS
python train.py -C src/configs/config_kon_all.yaml
```

------

## ✅ Validation & Downstream Tasks

You can validate the learned representations on tasks such as:

### 1. Enzyme classification

```bash
cd ProtDis/tasks/proteinshake
python tasks/train.py -C src/configs/ec/cdconv_structure.yaml
```

### 2. Protein-ligand binding site

```bash
cd ProtDis/tasks/proteinshake
python tasks/train.py -C src/configs/binding_site/cdconv_structure.yaml
```

------

## 📊 Results

Performance under **structure-based splits** across 12 downstream tasks:

| Task | Metric | ESM3ST | ProtDiS (ours) |
|------|--------|------|----------------|
| Enzyme Class (EC) | acc | 0.7869 | **0.8345** |
| GO-MF | fmax | 0.6112 | **0.6123** |
| GO-BP | fmax | 0.4026 | **0.4046** |
| GO-CC | fmax | 0.4623 | **0.4707** |
| Pfam | acc | 0.5885 | **0.5948** |
| SCOP-family | acc | 0.7502 | **0.7795** |
| SCOP-class | acc | 0.8614 | **0.8662** |
| SCOP-clan | acc | 0.9461 | **0.9601** |
| Structure Similarity | spearmanr | 0.6666 | **0.6692** |
| Ligand Affinity* | spearmanr | 0.3508 | **0.3664** |
| PPIs* | auroc | 0.8210 | **0.8461** |
| Ligand Binding Site | mcc | 0.6166 | **0.6232** |

------

## 📌 Future Work

- Pretrain protein knowledge language models
- Explore generative design of functional fragments

------

## 📬 Contact

For questions, please contact:

**Mingqing Wang**

 Ph.D. Student in Protein Representation Learning
 
 Email: wmq23@mails.tsinghua.edu.cn

------

## 📄 License


This project is licensed under the MIT License. See `LICENSE` for more details.
