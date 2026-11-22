# Deep Learning Coursework

Multimodal classification of disaster tweets and images with **CrisisMMD v2.0**. This repository contains the data preparation script, the shared environment, and (on this branch) the full **Part A** pipeline: models, training and evaluation, results, selected checkpoints and the environmental-impact run log. Each teammate downloads the data locally; images and cached features are not in Git.

## Repository structure

```text
DeepLearning-Coursework/
  PartA.ipynb                 Part A notebook: feature caching, training, evaluation, figures
  scripts/
    prepare_crisismmd.py      Data download & preparation
    crisis_models.py          Tokenizer, vocabulary, TextCNN/BiLSTM/image/fusion models, training, metrics
    run_logger.py             Per-run experiment + energy record (CodeCarbon, TensorBoard)
    energy_tensorboard.py     Streams live CodeCarbon energy into TensorBoard
    aggregate_runs.py         Builds the one-row-per-run log and report tables
    requirements-logging.txt  TensorBoard + CodeCarbon pins used by the logger
  templates/                  Run configurations + shared measurement method (carbon intensity, PUE, uncertainty)
  tests/                      Unit tests (python -m unittest discover -s tests)
  logs/
    run_log.csv               Experiment log, one row per run (environmental guidance, Section 3)
    stage_summary.csv         Energy/emissions by workflow stage (report Table 1)
    model_summary.csv         Final models: seeds, F1, energy (report Table 2)
    aggregate_summary.json    Run counts, totals with ranges, wasted-compute ratio
    runs/<experiment_id>/     Per run: config, summary, energy, epochs.csv, TensorBoard events, best.pt
  results/partA/              Result tables, per-example predictions, figures
  RUN_LOGGING_GUIDE.txt       How to log every run
  ENVIRONMENTAL_RUN_LOG_FIELDS.txt  Fields required by the environmental guidance
  crisismmd-cpu.yml / crisismmd-gpu.yml / requirements.txt   Shared group environment
  requirements-partA-cu128.txt      Exact environment used for the Part A runs
  data/                       Dataset and cached features (not in Git)
```

## Requirements

- Windows 10/11, macOS or Linux
- [Git](https://git-scm.com/downloads)
- **One of:** [Miniconda](https://www.anaconda.com/download/success) (scroll down to *Miniconda Installers*) for the conda method, **or** [Python 3.12](https://www.python.org/downloads/) for the venv method
- 8 GB RAM and about 10 GB free disk space (about 2 GB of it for the dataset)
- A GPU is **optional**: everything is designed to run on a CPU. The GPU environment needs an NVIDIA GPU whose driver supports CUDA 12.4 or newer.

## Environment setup

There are two ways to set up the environment. Both install Python 3.12 and the same pinned package versions, so **pick one**:

- **Method A: conda (recommended).** Conda installs Python 3.12 for you, so it works even if you have a different Python version installed.
- **Method B: venv.** Use this if you already have Python 3.12 installed and don't want conda.

### 1. Clone the repository

```bash
git clone https://github.com/MehulChourasia28/DeepLearning-Coursework.git
cd DeepLearning-Coursework
```

### 2. Choose CPU or GPU

Use **CPU** if you have no NVIDIA GPU, are on a Mac, or are unsure. The coursework is designed to run on a CPU, so a GPU is only a speed-up.

To check for an NVIDIA GPU, run `nvidia-smi`. An error means use **CPU**. A table means you have one: if **CUDA Version** (top right) is **12.4 or higher**, you can use **GPU**; otherwise update your NVIDIA driver or use CPU.

### 3A. Method A: conda

> **Windows:** type these commands into **Anaconda Prompt (miniconda3)** (press the Windows key and search for it). **macOS / Linux:** use a normal terminal.

| File | Environment name | Use it for |
| --- | --- | --- |
| `crisismmd-cpu.yml` | `crisismmd-cpu` | CPU (any OS) |
| `crisismmd-gpu.yml` | `crisismmd-gpu` | NVIDIA GPU (Windows/Linux) |

Create and activate the environment. This takes 5–15 minutes (the GPU version downloads about 2.5 GB). Run the pair that matches your choice:

```bash
conda env create -f crisismmd-cpu.yml
conda activate crisismmd-cpu
```

```bash
conda env create -f crisismmd-gpu.yml
conda activate crisismmd-gpu
```

Your command line should now start with `(crisismmd-cpu)` or `(crisismmd-gpu)`. **Run the `conda activate` line every time you open a new terminal.**

### 3B. Method B: venv

> **Windows:** use **Command Prompt**. **macOS / Linux:** use a normal terminal.

**Check that you have Python 3.12.** This should print `Python 3.12.x`:

```bash
py -3.12 --version
```

```bash
python3.12 --version
```

Use the `py -3.12` line on Windows and `python3.12` on macOS/Linux. If you get an error, install Python 3.12 from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** during installation.

**Create the virtual environment.** This makes a `.venv` folder inside the repo, which Git ignores. Use the first line on Windows and the second on macOS/Linux:

```bash
py -3.12 -m venv .venv
```

```bash
python3.12 -m venv .venv
```

**Activate it.** Use the line for your system:

| System | Command |
| --- | --- |
| Windows (Command Prompt) | `.venv\Scripts\activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Your command line should now start with `(.venv)`. **Activate it every time you open a new terminal.**

If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and try again (or just use Command Prompt).

**Install the packages.** Run the block that matches your choice. It takes 5–15 minutes.

CPU:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU (installs the CUDA build of PyTorch first, then everything else):

```bash
python -m pip install --upgrade pip
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Check the installation

With your environment activated (either method), run:

```bash
python --version
python -c "import torch, torchvision, codecarbon; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `Python 3.12.x`, then `2.5.1+cpu False` for CPU (macOS shows `2.5.1 False`; Linux may show `2.5.1+cu124 False`) or `2.5.1+cu124 True` for GPU. If GPU prints `False`, your NVIDIA driver is probably too old.

### Updating the environment

When someone adds or changes a package, they must update **all three** files: `crisismmd-cpu.yml`, `crisismmd-gpu.yml` and `requirements.txt`. Everyone else then pulls the changes and runs the line for their setup:

| Setup | Command |
| --- | --- |
| Conda, CPU | `conda env update -n crisismmd-cpu -f crisismmd-cpu.yml --prune` |
| Conda, GPU | `conda env update -n crisismmd-gpu -f crisismmd-gpu.yml --prune` |
| venv (CPU or GPU, environment activated) | `pip install -r requirements.txt` |

### Exact environment for the Part A runs (RTX 50-series / CUDA 12.8)

The logged Part A results were produced with a newer stack than the shared group pins: PyTorch 2.11 with CUDA 12.8, CodeCarbon 3.3.2 and TensorBoard 2.21. The RTX 5060 Ti (Blackwell) is not supported by the CUDA 12.4 builds above. The run logger (`scripts/run_logger.py`) has only been tested with CodeCarbon 3.3.2, and it needs TensorBoard, which the shared environment does not include. To reproduce the Part A runs exactly, use a Python 3.12 venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-partA-cu128.txt
```

> **Group action needed:** if we standardise on this stack, update all three shared files (`crisismmd-cpu.yml`, `crisismmd-gpu.yml`, `requirements.txt`) together, as described above.

## Get the dataset

With your environment activated (conda or venv; both include everything the preparation script needs), run:

```bash
python scripts/prepare_crisismmd.py --workers 12
```

The script downloads the official agreed-label TSVs and their **12,708 unique images**, prepares the targets, and checks sample counts and image decoding. It pins the author repository and image mirror revisions. Re-run the same command to resume an interrupted download; it waits automatically if the mirror rate-limits requests. No GPU or PyTorch is needed for preparation.

## Dataset folders

```text
data/crisismmd/
  data_image/            Images grouped by disaster and collection date
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

## Part A: reproduce the results

### Hardware and software used

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 7900 (12 cores / 24 threads) |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB (driver 580.178.04) |
| RAM | 32 GB (30.5 GB usable) |
| OS | Ubuntu 24.04, Linux 7.0 |
| Software | Python 3.12.3, PyTorch 2.11.0+cu128, CodeCarbon 3.3.2 (see `requirements-partA-cu128.txt`) |

Each run's `logs/runs/<id>/summary.json` records the exact hardware it ran on.

### Commands

From the repository root, with the dataset prepared (see *Get the dataset*) and the environment activated:

```bash
# Linux only, once per boot: let CodeCarbon read the CPU energy counter (RAPL),
# so CPU energy is measured rather than estimated. Resets on reboot.
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj

# Unit tests (seconds, CPU only)
python -m unittest discover -s tests

# Full run: open PartA.ipynb in VS Code (Jupyter extension, select the .venv kernel)
# or in Jupyter, then "Run All" (cells top to bottom). The shared group environment
# includes Jupyter; with requirements-partA-cu128.txt, install it first:
python -m pip install notebook
jupyter notebook PartA.ipynb

# CPU-only run: hide the GPU from the process that starts the kernel
CUDA_VISIBLE_DEVICES="" jupyter notebook PartA.ipynb    # or: CUDA_VISIBLE_DEVICES="" code .

# Rebuild the experiment log and report tables from all run records
python -m scripts.aggregate_runs
```

The code falls back to the CPU automatically when no GPU is available. On the GPU above the whole notebook takes about 3 minutes of compute. The slowest step is feature extraction (about 1 minute), which decodes all 12,708 images and runs ResNet18; on a CPU that step takes noticeably longer. Training uses the cached features, and each run takes 1–4 s on the GPU.

**Every execution creates new run records.** IDs get a timestamp and random suffix, and existing records are never overwritten. Re-running a cell is therefore always logged as a new run: keep it, and set `rerun_of`/`rerun_reason` in its configuration. The idle-baseline cell sleeps for 60 s to measure background power, so close TensorBoard and other heavy programs while it runs.

### Notebook order

1. Data loaders, then an **idle baseline** run (fixed background power).
2. **Debug** forward pass of frozen ResNet18 (one batch).
3. **Feature extraction:** frozen ImageNet ResNet18 (`IMAGENET1K_V1`, prescribed preprocessing, `fc` = identity), producing 512-d features cached to `data/crisismmd/features/informative_{train,valid,test}.pt`.
4. **Text preprocessing:** tokenizer, training-only vocabulary, 64-token limit (logged run).
5. **Pilot runs** (debug): 1 epoch on 1,024 training examples for each learned model.
6. **Final training:** majority baseline, plus text only (TextCNN), image only, fusion (TextCNN + image) and fusion (BiLSTM + image), each with seeds 0, 1 and 2.
7. **Median-seed selection** by validation macro-F1, saved before any test evaluation (`results/partA/median_seed_selection.json`).
8. **Final test evaluation:** a separate record per training run, linked by `evaluated_experiment_id`.
9. Results tables, the Ofli et al. (2020) accuracy comparison, learning curves, confusion matrices, fusion examples, and the environmental-impact figures and tables.

### Configuration

| Setting | Value |
| --- | --- |
| Image branch | cached frozen ResNet18 features (512) → Linear 256 + ReLU + dropout 0.5 |
| TextCNN | 128-d embeddings; kernels 2, 3, 4, 5 × 64 filters; max-pool that ignores padding → 256 |
| BiLSTM | 128-d embeddings; 1 bidirectional layer, 128 units per direction → 256 |
| Classifier head (shared by all models) | Linear(input → 128) + ReLU + dropout 0.5 → Linear(128 → 2) |
| Vocabulary | training tweets only, deduplicated by `tweet_id`; minimum frequency 2; 64-token limit |
| Tokenizer | lower-case; URLs → `<url>`; @mentions → `<user>`; `#` and a leading `RT` removed |
| Optimiser and loss | Adam, learning rate 1e-3, batch 64, unweighted cross-entropy, no LR scheduler |
| Epochs | at most 20; early stopping with patience 3 on validation macro-F1; best-epoch checkpoint kept |
| Seeds | 0, 1, 2 (`random`, NumPy, PyTorch and the shuffle generator; cuDNN deterministic mode) |
| Metrics | macro-F1 over both classes (0–1 scale), accuracy, informative precision/recall, confusion matrix |

The hyperparameters are in `scripts/crisis_models.py` (`HYPERPARAMETERS`), and each run's full configuration is saved in `logs/runs/<id>/config.json`.

### Selected checkpoints

Every final training run saves its validation-selected checkpoint as `logs/runs/<experiment_id>/best.pt`. The confusion-matrix run for each model is its median-validation seed (`results/partA/median_seed_selection.json`).

**Selected Part A checkpoint:** fusion (BiLSTM + image), seed 2: `logs/runs/A_fusion_bilstm_seed2_final_20261122T145925_aaa4d44d/best.pt`. Rule: the model with the highest mean validation macro-F1 (0.8510 against 0.8505 for TextCNN fusion, effectively a tie), taking that model's median-validation seed. No test data were used for this choice.

To load it:

```python
import json, torch
from scripts.crisis_models import build_model
vocab = json.load(open("results/partA/vocab.json"))
model = build_model("fusion_bilstm", vocab_size=len(vocab), num_classes=2)
model.load_state_dict(torch.load("logs/runs/A_fusion_bilstm_seed2_final_20261122T145925_aaa4d44d/best.pt")["model"])
```

### Outputs

| Path | Contents |
| --- | --- |
| `results/partA/results_per_run.csv` | Model, seed, validation macro-F1, selected epoch, test scores, trainable parameters, training time, energy |
| `results/partA/results_table.csv`, `summary_mean_std.csv` | Mean ± standard deviation over seeds |
| `results/partA/published_comparison.csv` | Accuracy against Ofli et al. (2020) |
| `results/partA/predictions/` | Per-example validation and test predictions with class probabilities, for every run |
| `results/partA/figures/` | Learning curves, confusion matrices, fusion examples, energy figures |
| `results/partA/vocab.json` | Training vocabulary (index = token id) |
| `logs/runs/<id>/epochs.csv` | Per-epoch losses, validation macro-F1 and learning rate (learning curves) |

## Environmental-impact logging

Every computational step is a logged run: debug, pilot, feature extraction, training and evaluation. Each run has a unique ID, run type and workflow stage. `scripts/run_logger.py` records runtime and CodeCarbon energy:
- CPU energy is measured with RAPL and GPU energy with NVML; RAM energy is estimated.
- Emissions use 122 gCO₂/kWh, the NESO Great Britain average for Nov 2025 to Oct 2026.
- Each run also records a documented low/high uncertainty range.

The shared assumptions are in `templates/measurement_method.json`. See `RUN_LOGGING_GUIDE.txt` for how to log new runs (including Parts B and C) and `ENVIRONMENTAL_RUN_LOG_FIELDS.txt` for the required fields.

```bash
# Live energy, loss and F1 curves
tensorboard --logdir logs/runs --port 6006
```

## Sources and licence

- Annotations come from the [authors' baseline repository](https://github.com/firojalam/multimodal_social_media); images come from [QCRI's CrisisMMD v2.0 mirror](https://huggingface.co/datasets/QCRI/CrisisMMD). Dataset licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Dataset images shown in `results/partA/figures/fusion_examples.png` are covered by the same licence.
- Cite [Alam et al. (2018)](https://arxiv.org/abs/1805.00713) (CrisisMMD) and [Ofli et al. (2020)](https://arxiv.org/abs/2004.11838) (published baselines).
- ResNet18 ImageNet weights and preprocessing: [torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- Energy tracking: [CodeCarbon](https://docs.codecarbon.io/); carbon intensity: [NESO Carbon Intensity API](https://api.carbonintensity.org.uk/).
