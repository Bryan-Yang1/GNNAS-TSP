# GNNAS-TSP: Graph Neural Network Based Algorithm Selection for the Traveling Salesman Problem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GNNAS-TSP is a graph neural network based framework for algorithm selection on Traveling Salesman Problem (TSP) instances. Given a TSP instance, the model predicts solver performance and selects among five candidate solvers: CLK, EAX, LKH, MAOS, and CONCORDE. This is the framework we used in our model. Other solvers can also be applied if needed.

This repository contains the training entry point, experiment split files, hyperparameter sensitivity launcher, and result-analysis notebooks used for the Summer Research Scholarship (SRS) project at Duke Kunshan University.

Our research paper will be published here soon!

---

## Repository Layout

```text
GNNAS-TSP/
|
|-- main.py                         # Training / evaluation entry point
|-- main.sh                         # Preset shell runner for main.py
|-- hyperparameter_sensitivity.py   # One-factor sensitivity launcher
|
|-- data/
|   |-- *.json                      # Instance ID split/filter files
|   `-- README.md                   # Full dataset download instructions
|
|-- results_analysis/
|   |-- summarize_general_selectors.ipynb
|   |-- analyze_best_selector_vbs_gap_wilcoxon.ipynb
|   `-- README.md
|
|-- environment.yaml                # Conda environment specification
|-- requirements.txt                # pip dependency list
|-- .gitignore
`-- README.md
```

Generated experiment outputs, model checkpoints, full TSP instances, and solver execution logs are intentionally not tracked by Git.

---

## Installation

Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate gnnaser
```

Or install with pip:

```bash
pip install -r requirements.txt
```

For CUDA machines, install the PyTorch build that matches the server before running experiments. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Data Preparation

The `data/` directory in this repository contains JSON files with instance IDs used for train/test splits and filtering. These files do not contain the full TSP dataset.

Download the full TSP instances and solver execution logs from Google Drive:

https://drive.google.com/drive/folders/1RABzr-qS9OKF5Za1G1IUxUO5WgjK-BPq?usp=sharing

After downloading, place or symlink the full dataset under `data/`:

```text
data/
|-- tsp_instances_training/
|-- tsp_instances_testing/
|-- tsp_executions/
|-- filtered_ids_*.json
`-- ...
```

Expected solver execution layout:

```text
data/tsp_executions/
|-- CLK/
|-- EAX/
|-- LKH/
|-- MAOS/
`-- CONCORDE/
```

Each solver directory should contain the `.out` files used to extract cost and runtime labels.

---

## Quick Start

Run the default training preset:

```bash
bash main.sh
```

The preset uses:

- `TIME_LIMIT=60`
- `DATA_ROOT=data`
- `RESULTS_ROOT=outputs`
- `DEVICE=auto`

You can override these without editing the script:

```bash
TIME_LIMIT=10 DATA_ROOT=data RESULTS_ROOT=outputs DEVICE=cuda bash main.sh
```

For a direct Python run:

```bash
python main.py \
  --time_limit 10 \
  --data_root data \
  --results_root outputs \
  --device auto \
  --hidden_dim 128 \
  --num_layers 3 \
  --mlp_layers 2 \
  --learning_rate 0.001 \
  --decay_rate 1.2 \
  --max_epochs 30 \
  --rank_method dense \
  --cost_loss MSE \
  --rank_loss ListNet \
  --loss_weight 0.5
```

To run on a server where the dataset is stored elsewhere:

```bash
python main.py \
  --data_root /path/to/GNNASeR/data \
  --results_root /path/to/GNNASeR/outputs \
  --device cuda
```

---

## Hyperparameter Sensitivity

The sensitivity launcher performs one-factor-at-a-time runs for:

- `decay_rate`: `1.0`, `1.2`, `1.5`
- `max_epochs`: `20`, `30`, `40`

The base configuration keeps:

- `hidden_dim=128`
- `num_layers=3`
- `mlp_layers=2`
- `learning_rate=0.001`

Preview generated commands:

```bash
python hyperparameter_sensitivity.py
```

Execute the runs:

```bash
python hyperparameter_sensitivity.py --execute
```

Server example:

```bash
python hyperparameter_sensitivity.py \
  --execute \
  --data_root /path/to/GNNASeR/data \
  --results_root /path/to/GNNASeR/hparam_sensitivity \
  --device cuda
```

---

## Results Analysis

Analysis notebooks are stored in `results_analysis/`:

- `summarize_general_selectors.ipynb`: builds summary tables for selectors, fixed solvers, SBS, and VBS.
- `analyze_best_selector_vbs_gap_wilcoxon.ipynb`: computes VBS gaps and Wilcoxon signed-rank tests for best-selector comparisons.

The notebooks expect generated result folders such as `10s_v2/`, `60s_v2/`, or similar experiment outputs. If your result folders are stored elsewhere, update the `RESULTS_ROOT` configuration cell in the notebook.

---

## Important Arguments

- `--time_limit`: Solver time cutoff used when extracting labels from solver logs.
- `--data_root`: Root directory containing `tsp_instances_training/`, `tsp_instances_testing/`, and `tsp_executions/`.
- `--train_ids_json`, `--test_ids_json`: Optional JSON files limiting the train/test instance IDs.
- `--num_neighbors`: Graph construction setting; `-1` uses a fully connected graph.
- `--hidden_dim`, `--num_layers`, `--mlp_layers`: Model architecture settings.
- `--cost_loss`: Cost regression loss, one of `MSE`, `MAE`, or `Huber`.
- `--rank_loss`: Ranking loss, one of `RankNet`, `ListNet`, or `LambdaRank`.
- `--loss_weight`: Weight assigned to the cost loss; `1 - loss_weight` is assigned to the ranking loss.
- `--device`: `auto`, `cuda`, `mps`, or `cpu`.
- `--results_root`: Directory where checkpoints, predictions, and loss curves are written.

---

## Acknowledgements

This project was developed as part of the Summer Research Scholarship (SRS) at Duke Kunshan University. GNNASeR builds on PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, and Matplotlib.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
