# Deep Learning Coursework

Download and prepare **CrisisMMD v2.0** for the compulsory tweet–image classification tasks. This repository contains the preparation script and instructions; each teammate downloads the data locally.

## Repository structure

```text
DeepLearning-Coursework/
  data/                     Dataset and cached features (not in Git)
  crisismmd-cpu.yml         Conda environment for CPU (any OS)
  crisismmd-gpu.yml         Conda environment for NVIDIA GPU (Windows/Linux)
  requirements.txt          Packages for the venv method (same versions)
  scripts/                
    prepare_crisismmd.py    Data download & preparation
  
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