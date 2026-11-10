# Deep Learning Coursework

Download and prepare **CrisisMMD v2.0** for the compulsory tweet–image classification tasks. This repository contains the preparation script and instructions; each teammate downloads the data locally.

## Get the dataset

Use Python 3.12 and allow about 2 GB of disk space for the dataset.

```bash
git clone https://github.com/MehulChourasia28/DeepLearning-Coursework.git
cd DeepLearning-Coursework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-data.txt
python scripts/prepare_crisismmd.py --workers 12
```

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

## Sources

[Official annotations and baseline](https://github.com/firojalam/multimodal_social_media) · [QCRI image mirror and CC BY-NC-SA 4.0 licence](https://huggingface.co/datasets/QCRI/CrisisMMD). Cite [Alam et al. (2018)](https://arxiv.org/abs/1805.00713) and [Ofli et al. (2020)](https://arxiv.org/abs/2004.11838).
