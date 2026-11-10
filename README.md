# Deep Learning Coursework

Download and prepare **CrisisMMD v2.0** for the compulsory tweet–image classification tasks. This repository contains the preparation script and instructions; each teammate downloads the data locally.

## Requirements

- Windows 10/11, macOS or Linux
- [Miniconda](https://www.anaconda.com/download/success) (scroll down to *Miniconda Installers*) and [Git](https://git-scm.com/downloads)
- 8 GB RAM and about 10 GB free disk space (about 2 GB of it for the dataset)
- A GPU is **optional**: everything is designed to run on a CPU. The GPU environment needs an NVIDIA GPU whose driver supports CUDA 12.4 or newer.

## Environment setup

> **Windows:** type the commands into **Anaconda Prompt (miniconda3)** (press the Windows key and search for it).
> **macOS / Linux:** use a normal terminal.

### 1. Clone the repository

```bash
git clone https://github.com/MehulChourasia28/DeepLearning-Coursework.git
cd DeepLearning-Coursework
```

### 2. Choose CPU or GPU

There are two conda environment files. Both install Python 3.12 and the same pinned package versions; only the PyTorch build differs.

| File | Environment name | Use it if… |
| --- | --- | --- |
| `crisismmd-cpu.yml` | `crisismmd-cpu` | you have no NVIDIA GPU or are on a Mac. **If unsure, use this one.** |
| `crisismmd-gpu.yml` | `crisismmd-gpu` | you have an NVIDIA GPU on Windows or Linux |

To check for an NVIDIA GPU, run `nvidia-smi`. An error means use the **CPU** file. A table means you have one: if **CUDA Version** (top right) is **12.4 or higher**, use the **GPU** file; otherwise update your NVIDIA driver or use the CPU file.

### 3. Create and activate the environment

This takes 5–15 minutes (the GPU version downloads about 2.5 GB). Run the pair that matches your choice:

```bash
conda env create -f crisismmd-cpu.yml
conda activate crisismmd-cpu
```

```bash
conda env create -f crisismmd-gpu.yml
conda activate crisismmd-gpu
```

Your command line should now start with `(crisismmd-cpu)` or `(crisismmd-gpu)`. **Activate the environment every time you open a new terminal.**

### 4. Check the installation

```bash
python --version
python -c "import torch, torchvision, codecarbon; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `Python 3.12.x`, then `2.5.1+cpu False` for CPU (macOS shows `2.5.1 False`; Linux may show `2.5.1+cu124 False`) or `2.5.1+cu124 True` for GPU. If the GPU environment prints `False`, your NVIDIA driver is probably too old.

Optional sanity check: `cd examples`, `python example_1.py`, `cd ..`. The first run downloads a pretrained object detector; it should print `Done!` and create `examples/street_bboxes.jpg` with boxes around cars and people.

### Updating the environment

When someone adds a package, add it to **both** `crisismmd-cpu.yml` and `crisismmd-gpu.yml`. Everyone else then runs the line for their environment:

```bash
conda env update -n crisismmd-cpu -f crisismmd-cpu.yml --prune
```

```bash
conda env update -n crisismmd-gpu -f crisismmd-gpu.yml --prune
```

## Get the dataset

With your conda environment activated (it already includes everything the preparation script needs), run:

```bash
python scripts/prepare_crisismmd.py --workers 12
```

*Data-only alternative (no conda, no PyTorch):* create a Python 3.12 virtual environment and run `python -m pip install -r scripts/requirements-data.txt` before the command above.

The script downloads the official agreed-label TSVs and their **12,708 unique images**, prepares the targets, and checks sample counts and image decoding. It pins the author repository and image mirror revisions. Re-run the same command to resume an interrupted download; it waits automatically if the mirror rate-limits requests. No GPU or PyTorch is needed for preparation.

## Generated folders

```text
data/crisismmd/
  data_image/             Images grouped by disaster and collection date
  splits/                Six unchanged official TSVs (dev = validation)
  prepared/
    informative/         Part A: train.tsv, valid.tsv, test.tsv
    humanitarian/        Part B: train.tsv, valid.tsv, test.tsv
```

| Task | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| A — Informativeness (2 classes) | 9,601 | 1,573 | 1,534 |
| B — Humanitarian (5 classes) | 6,126 | 998 | 955 |

Use **`prepared/` for modelling** and predict `label`. Rows contain tweet text, event, IDs and an image path relative to `data/crisismmd/`. Keep `tweet_id` and `image_id` as strings. Official split membership and row order are preserved.

For Part B, injured/dead and missing/found labels map to `affected_individuals`; vehicle damage maps to `infrastructure_and_utility_damage`. This matches the authors' five-class agreed labels. `label_original` retains the source value.

To recheck local files without downloading:

```bash
python scripts/prepare_crisismmd.py --verify-only
```

Generated data and local JSON verification reports are excluded from Git. Optional Part C3 all-label data is not downloaded.

## Repository structure

```text
DeepLearning-Coursework/
  crisismmd-cpu.yml       Conda environment for CPU (any OS)
  crisismmd-gpu.yml       Conda environment for NVIDIA GPU (Windows/Linux)
  requirements-lock.txt   Exact installed package versions (pip freeze)
  configs/                One YAML config per experiment
  data/                   Dataset and cached features (not in Git)
  src/crisismmd/          Reusable code: data, models, training, metrics, tracking
  scripts/                Entry points (data preparation, Parts A, B and C)
  notebooks/              Error analysis and Matplotlib figures
  examples/               example_1.py + street.jpg (environment sanity check)
  results/
    logs/                 experiment_log.csv + CodeCarbon emissions
    metrics/              CSV/JSON scores per model and seed
    predictions/          Per-example predictions and probabilities
    figures/              Learning curves, confusion matrices, energy plots
    checkpoints/          Selected model weights (large files not in Git)
  report/                 Environmental impact section and tables
```

## Reproducibility

- Final learned models use **seeds 0, 1 and 2**; results report mean ± standard deviation.
- Checkpoints are selected by **validation macro-F1** with early-stopping patience 3; test data is used only for final evaluation.
- Package versions are pinned in the environment files; exact installed versions are in `requirements-lock.txt`.
- Every run (debug, exploratory, final or failed) gets an experiment ID and a row in `results/logs/experiment_log.csv`, with runtime, energy (kWh) and emissions (kgCO₂e) measured by CodeCarbon, plus the hardware used.

### Hardware used

| Member | Device | CPU | GPU | RAM | OS |
| --- | --- | --- | --- | --- | --- |
| *name* | *laptop/desktop* | *e.g. Intel i7-1260P* | *none / RTX 3060* | *16 GB* | *Windows 11* |

## Team

| Member | Contributions |
| --- | --- |
| *name* | *to be added* |

## Sources

[Official annotations and baseline](https://github.com/firojalam/multimodal_social_media) · [QCRI image mirror and CC BY-NC-SA 4.0 licence](https://huggingface.co/datasets/QCRI/CrisisMMD). Cite [Alam et al. (2018)](https://arxiv.org/abs/1805.00713) and [Ofli et al. (2020)](https://arxiv.org/abs/2004.11838).
