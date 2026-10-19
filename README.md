# PDAttn: Perturbation-Specific Expression Prediction

## 1. Overview

**PDAttn** predicts gene expression under genetic perturbations in single-cell settings. It uses graph structure related to perturbations (including Gene Ontology, GO) and integrates perturbation signals at **two levels**:

- **Level 1 — gene-level conditioning:** perturbation features **modulate** gene representations (FiLM-style conditioning) so the model encodes **perturbation-aligned, gene-resolution** responses.
- **Level 2 — context-aware injection:** perturbation features are combined with **cell background** information to form a context vector, which is injected into gene representations to capture **perturbation effects that vary across cellular contexts**.

Place the architecture figure at the path below so it renders on GitHub and locally:

![PDAttn architecture](img/PDAttn%20Model.png)

For splits and PyG dataloaders, this codebase follows conventions aligned with **GEARS**, enabling fair comparison with existing perturbation baselines: [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS).

---

## 2. Requirements

- **Python:** 3.9 or newer recommended.
- **Dependencies:** see [`requirements.txt`](requirements.txt) in the repository root.

**Suggested setup** (run from the repository root, e.g. `PDAttn-main`):

```bash
cd PDAttn-main
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
```

For **GPU** runs, install a CUDA-matched **PyTorch** build first ([PyTorch — Get Started](https://pytorch.org/get-started/locally/)), then install the rest:

```bash
pip install -r requirements.txt
```

**CPU-only** installs can use the same command:

```bash
pip install -r requirements.txt
```

If `torch-geometric` fails to install, follow [PyTorch Geometric — Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and pick the wheel index matching your PyTorch and CUDA version.

---

## 3. Paper configuration and complete hyperparameter specification

This section is the authoritative configuration record for reproducing the paper. The
same model and optimization configuration was used for **all eight datasets**; there
was no dataset-specific tuning. The symbols in the manuscript map to implementation
names as follows.

| Manuscript symbol / term | Implementation name | Paper value | Unit | Meaning |
|---|---|---:|---|---|
| Latent dimension \(d\) | `hidden_dim`, `hidden_size`, `PDAttnModel.d` | 64 | features | Width of the scalar-expression projection, static gene embedding, perturbation embedding, FiLM representation, and attention input/output |
| Number of attention heads \(H\) | `num_attention_heads` | 4 | heads | Parallel heads in both gene self-attention and perturbation–cell cross-attention |
| Per-head width \(d_k=d/H\) | `LinearAttention.d_k` | 16 | features/head | Query, key, and value width assigned to each attention head |
| Dropout probability | `dropout` | 0.1 | probability | Applied in attention, FiLM, perturbation–cell fusion, and combination-residual modules |
| GO neighborhood size | `k_go` | 20 | neighbors/node | Nearest GO-related perturbation neighbors retained during graph construction |
| GO message-passing depth | `num_go_gnn_layers` | 1 | SGConv layer | Number of perturbation-side SGConv layers; each uses `K=1` propagation step |
| CDF loss weight \(\lambda_{CDF}\) | `SystemaLoss.cdf_lambda` | 0.2 | dimensionless | Multiplier on the empirical-distribution matching term |
| CDF difference exponent | `SystemaLoss.cdf_p` | 2 | dimensionless | Exponent applied to absolute differences between sorted predicted and observed expression changes |
| Combination residual scale | `combo_de_scale` | 0.1 | dimensionless initial value | Learnable initial multiplier for the double-perturbation residual head |
| Decoder compatibility field | `decoder_hidden_size` | 16 | features | Serialized for checkpoint/API compatibility; the current residual output head does not instantiate a separate decoder layer from this value |

The linear-attention feature map is \(\phi(x)=\mathrm{ELU}(x)+1\). Both attention
modules use four heads with 16 features per head. The cell compressor has widths
`G -> 128 -> 64`, where `G` is the number of retained genes. The perturbation–cell
fusion layer has widths `128 -> 64`; the FiLM scale and shift networks each have
widths `64 -> 64 -> 64`; and the combination-specific residual head has widths
`64 -> 64 -> 1`. The perturbation graph uses one `SGConv(64, 64, K=1)` layer.

### 3.1 Data split and loader configuration

| Setting | Paper value | Unit | Reproducibility note |
|---|---:|---|---|
| Dataset split | `simulation` | categorical | GEARS-style simulation split used by training and evaluation |
| `train_gene_set_size` | 0.75 | fraction | Fraction of perturbation genes assigned to the training gene set |
| Random seed | 42 | integer seed | Used for the split, parameter initialization, NumPy, and PyTorch RNGs |
| Training batch size | 128 | cells/batch | Training loader uses `shuffle=True` and `drop_last=True` |
| Validation/test batch size | 128 | cells/batch | Test loader uses `shuffle=False` |
| Cell graph input precision | FP32 | bit width | Default PyTorch floating-point execution; AMP is not enabled |
| DataLoader workers | 0 (PyG default) | worker processes | Data loading occurs in the main process unless explicitly changed |

`coexpr_threshold=0.4` is passed through the shared graph-builder interface. The
paper model calls `network_type="go"`; therefore this threshold is not used to prune
the GO branch and should not be interpreted as an additional tuned model parameter.

### 3.2 Optimization and loss configuration

| Setting | Paper value | Unit | Implementation detail |
|---|---:|---|---|
| Optimizer | AdamW | categorical | `torch.optim.AdamW` |
| Learning rate | \(1\times10^{-4}\) | dimensionless | Constant learning rate; no scheduler |
| Weight decay | \(1\times10^{-5}\) | dimensionless | Decoupled AdamW weight decay |
| Adam coefficients | \(\beta_1=0.9,\ \beta_2=0.999\) | dimensionless | PyTorch AdamW defaults |
| Adam numerical epsilon | \(1\times10^{-8}\) | dimensionless | PyTorch AdamW default |
| Training epochs | 15 | epochs | Fixed budget; no early stopping |
| Gradient clipping | none | not applicable | Gradients are not clipped |
| Learning-rate warm-up | none | not applicable | No learning-rate warm-up |
| Mixed precision | disabled | binary | Training and inference use FP32 |
| MSE term | enabled | binary | Mean squared error between absolute predicted and observed expression |
| CDF term | enabled | binary | Squared difference between sorted predicted and observed expression changes relative to the control centroid |
| CDF loss weight | 0.2 | dimensionless | Total loss = MSE + 0.2 × CDF term |

The control centroid `C_ctrl` is estimated from all control cells available in the
training loader. The main training script evaluates the validation split and overwrites
the checkpoint after every epoch; the released paper checkpoint is therefore the
**final (epoch 15) checkpoint**, not a validation-selected early-stopping checkpoint.

### 3.3 Parameter selection and sensitivity

The reproducibility configuration is a **single globally fixed configuration**:
`d=64`, `H=4`, `k_go=20`, and `lambda_CDF=0.2` are applied unchanged to every
dataset. The released training command does not perform a dataset-specific grid search
and does not select different hyperparameters from validation data. This distinction is
important: the values above reproduce the reported model, whereas a sensitivity study
is a separate experiment and must state its candidate set and selection criterion
explicitly. The repository does not claim candidate values or sensitivity results that
were not actually run.

For future sensitivity analyses, each candidate must retain the same split and seed and
should be compared using one prespecified validation statistic (for example, validation
PCC on DE20 genes). Test data must not be used for hyperparameter selection.

### 3.4 Training CLI — `task/train_model.py`

| Argument | Default | Unit | Description |
|---|---:|---|---|
| `--data_dir` | `./data` | path | Root directory passed to `PertData` |
| `--dataset` | `norman` | dataset key | Comma-separated names train multiple datasets sequentially |
| `--epochs` | 15 | epochs | Number of optimization epochs per dataset |
| `--lr` | `1e-4` | dimensionless | AdamW learning rate |
| `--batch_size` | 128 | cells/batch | Training batch size |
| `--test_batch_size` | 128 | cells/batch | Validation/test batch size |
| `--seed` | 42 | integer seed | Split and training seed |
| `--gpu` | `0` | device identifier | One GPU, comma-separated GPUs, or `cpu` |

Checkpoints are written to `model/<dataset_name>.pt` and contain the model state,
`C_ctrl`, optional assessment reference tensors, command arguments, and the last
validation metrics.

### 3.5 Evaluation CLI — `task/analysis_pearson.py`

| Argument | Default | Unit | Notes |
|---|---:|---|---|
| `--split` | `simulation` | categorical | Must match training |
| `--seed` | 42 | integer seed | Must match training |
| `--batch_size` | 128 | cells/batch | Evaluation batch size |
| `train_gene_set_size` | 0.75 | fraction | Fixed in the evaluation script; must match training |
| `--ckpt` | `model/<dataset>.pt` | path | Optional explicit checkpoint path |

The `override_*` options are intended only for loading checkpoints whose stored
architecture differs from the paper defaults. Do not use them for the paper checkpoints.

### 3.6 Optional scripts

`task/data_load.py` performs bulk preprocessing with its own historical defaults
(`seed=1`, training batch size 32). These values do **not** define the paper split;
the paper training/evaluation commands below recreate the `seed=42` split. The optional
linear baseline in `task/gslm.py` uses `pca_dim=64` and ridge `alpha=0.1`.

---

## 4. Datasets

We use public perturbation single-cell datasets; keys match `PertData.load`. NCBI GEO accessions and sources are listed below (links are clickable).

| `data_name` in code | Reference | Link |
|---------------------|-----------|------|
| `norman` | Norman et al., **GSE133344** | [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| `dixit` | Dixit et al., **GSE90063** | [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) |
| `adamson` | Adamson et al., **GSE90546** | [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) |
| `replogle_k562_essential` | Replogle et al., **GSE146194** (K562 essentials subset) | [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194) |
| `replogle_rpe1_essential` | Replogle et al., **GSE146194** (RPE1 essentials subset) | [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194) |
| `xu_kinetics_2024` | Xu et al., **GSE218566** | [GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566) |
| `tiankampmann2021_crispra` / `tiankampmann2021_crispri` | Tian / Kampmann et al. (CRISPRa / CRISPRi) | [Zenodo 13350497](https://zenodo.org/records/13350497) |

GEO series landing pages: [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344), [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063), [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546), [GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194), [GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566).

---

## 5. Repository layout and usage

Run the commands below from the repository root so that the `PDAttn`, `data_tools`, `tools`, and `task` packages resolve (or add the root to `PYTHONPATH`).

| Path | Role |
|------|------|
| `data_tools/PertData.py` | Data loading, GEARS-style splits, PyG `DataLoader` |
| `task/data_load.py` | (Optional) bulk download / preprocessing, splits, dataloaders |
| `task/train_model.py` | **Train** PDAttn; saves `model/<dataset>.pt` |
| `task/analysis_pearson.py` | **Inference and test-set evaluation** (Pearson, subgroups, top-DE, Systema-related metrics, etc.) |
| `task/gslm.py` | (Optional) gene-specific linear baseline — train and evaluate |

**Examples:**

```bash
# Reproduce one complete paper benchmark: Tian (CRISPRi).
python -m task.train_model \
  --dataset tiankampmann2021_crispri \
  --data_dir ./data \
  --gpu 0 \
  --epochs 15 \
  --batch_size 128 \
  --test_batch_size 128 \
  --lr 1e-4 \
  --seed 42

# Evaluate the matching test split.
python -m task.analysis_pearson \
  --dataset tiankampmann2021_crispri \
  --data_dir ./data \
  --ckpt model/tiankampmann2021_crispri.pt \
  --gpu 0 \
  --batch_size 128 \
  --seed 42 \
  --split simulation

# Train all eight paper datasets sequentially with the same configuration.
python -m task.train_model \
  --dataset norman,dixit,adamson,xu_kinetics_2024,replogle_k562_essential,replogle_rpe1_essential,tiankampmann2021_crispra,tiankampmann2021_crispri \
  --data_dir ./data \
  --gpu 0 \
  --epochs 15 \
  --batch_size 128 \
  --test_batch_size 128 \
  --lr 1e-4 \
  --seed 42
```

From the repository root, `python -m task.<module>` requires this directory on `PYTHONPATH` (the current working directory is enough when you `cd` here). Nested packages (`PDAttn`, `data_tools`, `tools`) do not ship `__init__.py` files; this is valid on **Python 3.3+** via [namespace packages](https://docs.python.org/3/reference/import.html#namespace-packages).

---

## 6. Processed data and direct reproduction

The processed datasets used by the paper are archived at:

> **Zenodo record 20064970:** <https://zenodo.org/records/20064970>

Download the processed-data archive and extract it under `./data` while preserving
the dataset directory names. The loader expects the following layout for each dataset:

```text
data/
├── gene2go_all.pkl
├── essential_all_data_pert_genes.pkl
└── tiankampmann2021_crispri/
    ├── perturb_processed.h5ad
    └── data_pyg/
        └── cell_graphs.pkl
```

The same pattern applies to the other seven dataset keys listed in Section 4. The
`perturb_processed.h5ad` files contain the processed expression matrices and metadata;
`data_pyg/cell_graphs.pkl` contains the corresponding preconstructed cell graphs. The
two root-level pickle files provide the GO annotations and default perturbation-gene
universe used by graph construction (the loader downloads them when they are absent).
The training command recreates the paper's `simulation` split with seed 42. Do not combine
checkpoints and evaluation splits generated with different seeds.

For provenance, record both the Zenodo record identifier and the exact source revision:

```bash
git rev-parse HEAD
python --version
python -c "import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch_geometric.__version__)"
pip freeze > environment-used.txt
```

The dependency ranges are declared in [`requirements.txt`](requirements.txt). For a
paper reproduction run, retain `environment-used.txt` together with the checkpoint and
command log so that the resolved package versions are auditable.

## 7. Versioned releases, figure scripts, and license status

- **Versioning.** Manuscript results should cite an immutable tagged release or commit
  SHA, not a moving branch. Create the publication tag only after the code, commands,
  processed-data link, and figure scripts have been frozen; archive that exact tag with
  the manuscript release.
- **Evaluation and figure provenance.** `task/analysis_pearson.py` produces the numerical
  evaluation outputs. Every manuscript plotting script should be included in the tagged
  publication release and should consume saved numerical outputs rather than rerunning
  model inference. Figure filenames and their generating commands should be listed in
  the release notes.
- **License.** PDAttn is distributed under the [MIT License](LICENSE), matching the
  license published with the public GitHub repository. Retain the license file in every
  tagged release and archived source package.

---

## Citation

A BibTeX entry and DOI will be added here after the manuscript is published.

---

## 8. Acknowledgements

We thank the authors and curators of the public datasets cited above (Norman, Dixit, Adamson, Replogle, Xu, Tian / Kampmann, and colleagues), and platforms such as GEO and Zenodo for open access. We also acknowledge community efforts such as **GEARS** for standardized splits and data practices that facilitate comparison and reproduction.
