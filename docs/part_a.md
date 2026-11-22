# Part A: Informativeness (binary) — reproduction guide and results

Predict whether a tweet–image pair is `informative` or `not_informative` (CrisisMMD v2.0, official
agreed-label splits). This document lists the exact commands, configuration, seeds, hardware and outputs
needed to reproduce every Part A number.

## 1. Pipeline at a glance

```
prepared TSVs ──► text: clean → tokenise → train-only vocab → 128-d embeddings → TextCNN | BiLSTM ──► 300-d ─┐
                                                                                                         concat ─► MLP ─► 2 classes
images ──► frozen ImageNet ResNet18 (eval mode) ──► cached 512-d features ───────────────────────────────┘
```

| Component | File |
|---|---|
| Data loading (IDs kept as strings), image-feature lookup | `crisismmd/data.py` |
| Text cleaning, tokenisation, training-only vocabulary | `crisismmd/text.py` |
| TextCNN, BiLSTM, image-only / text-only / fusion classifiers | `crisismmd/models.py` |
| Training loop, early stopping, prediction | `crisismmd/engine.py` |
| Metrics (macro/weighted F1, per-class, confusion matrix) | `crisismmd/metrics.py` |
| Experiment IDs, run log, energy tracking | `crisismmd/tracking.py` |
| Figures | `crisismmd/plotting.py` |
| Feature cache / training / summary / checkpoint evaluation | `scripts/extract_image_features.py`, `scripts/train.py`, `scripts/summarise.py`, `scripts/evaluate_checkpoint.py` |

## 2. Exact commands

Run from the repository root with the environment activated (see the main README). Everything runs on a CPU.

```bash
# 1. Data: official splits + images, verified (≈1.4 GB, not in Git)
python scripts/prepare_crisismmd.py --workers 12

# 2. Image features: frozen ResNet18, once for all 12,708 images (≈3 min on CPU)
python scripts/extract_image_features.py

# 3. Experiment-ID prefix (each group member uses their own)
export CRISISMMD_MEMBER=SHC

# 4. Idle baseline: 60 s of the idle machine (fixed background power, for the energy analysis)
python scripts/idle_baseline.py

# 5. Final training: majority baseline once (deterministic), five learned models × seeds 0, 1, 2 (≈7 min on CPU).
#    Training never touches the test split.
python scripts/train.py --task informative --model majority --run-type final --purpose "Part A final"
for seed in 0 1 2; do
  for m in "text --text-encoder textcnn" "text --text-encoder bilstm" "image" "fusion --text-encoder textcnn" "fusion --text-encoder bilstm"; do
    python scripts/train.py --task informative --model $m --seed $seed --run-type final --purpose "Part A final"
  done
done

# 6. Record the median-validation selection (timestamped), THEN evaluate every final checkpoint on test
python scripts/evaluate_final.py --task informative

# 7. Run log + Table 1/Table 2, then results tables/figures and energy figures
python -m scripts.aggregate_runs
python scripts/summarise.py --task informative --run-type final
python scripts/energy_figures.py --part A

# 8. Re-evaluate the submitted checkpoint without retraining
python scripts/evaluate_checkpoint.py checkpoints/part_a_informative_fusion_textcnn_SHC-A-026.pt --split test
```

GPU: add `--device cuda` to steps 2 and 5 (picked automatically when available). Quick pipeline check:
`--run-type debug --fraction 0.1 --max-epochs 2`.

**History note.** The reported Part A runs (`SHC-A-001`…`027`) were made with the first version of `train.py`, which
evaluated the test split inside each final run and logged to `results/runs/`. The pipeline was then moved to the
group run logger (`scripts/run_logger.py`) with test evaluation as a separate step (steps 5–6 above). The old
records were converted into the group format without retraining (`scripts/convert_legacy_runs.py`), and a
re-run of `SHC-A-010` with the new pipeline reproduced it exactly at every epoch.

## 3. Configuration

All values are the defaults of `scripts/train.py`; no hyperparameters were tuned.

| Setting | Value |
|---|---|
| Image encoder | torchvision ResNet18, `IMAGENET1K_V1` weights, `fc` removed, frozen, eval mode; weights' own preprocessing (resize 256, centre crop 224, ImageNet normalisation) → 512-d |
| Text preprocessing | repair mis-decoded UTF-8 where possible, unescape HTML, drop leading `RT`, URLs → `<url>`, `@user` → `<user>`, strip `#`, lowercase; tokens = words (any script) and `!` `?` |
| Vocabulary | training split only, tokens with count ≥ 2 (6,732 tokens incl. `<pad>`, `<unk>`) |
| Max tokens | 64 (Part A tweets have at most 26 whitespace-separated words and at most 32 tokens after tokenisation, so nothing is truncated) |
| Embeddings | 128-d, learned from scratch |
| TextCNN | kernel sizes 3/4/5 × 100 filters, ReLU, max-pool over valid positions → 300-d |
| BiLSTM | 1 layer, 150 hidden units per direction, final forward + backward states → 300-d |
| Classifier head (all models) | Dropout 0.5 → Linear(·, 256) → ReLU → Dropout 0.5 → Linear(256, 2) |
| Fusion | concatenate [image 512, text 300] → head. Both text encoders output 300-d, so the image branch and head are identical in the TextCNN ↔ BiLSTM swap |
| Optimiser / loss | Adam, lr 1e-3, no weight decay; cross-entropy |
| Batch size / epochs | 64 / at most 20 |
| Model selection | best epoch by **validation macro-F1**; early stopping after 3 epochs without improvement |
| Seeds | 0, 1, 2 (Python, NumPy, PyTorch, data order); deterministic algorithms enabled |
| Test split | evaluated only in `final` runs (enforced by `train.py`) |
| Confusion matrices / examples | from the seed with the **median validation** macro-F1 per model, chosen before inspecting test |

## 4. Hardware and software used

| | |
|---|---|
| CPU | Intel Core i5-13500H (16 threads; PyTorch used 12), no GPU |
| RAM | 15.2 GiB |
| OS | Linux 6.8 (glibc 2.39) |
| Software | Python 3.12.3, torch 2.10.0+cpu, torchvision 0.25.0+cpu, numpy 2.2.6, pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.8, Pillow 12.1.1, codecarbon 3.3.2 |

These versions are newer than the repository pins (torch 2.5.1 etc.); the code runs on both, but exact
numbers can differ slightly between library versions and hardware.

**Determinism check:** the seed-0 final runs reproduced the earlier exploratory runs with the same
settings exactly (identical loss and macro-F1 at every epoch).

## 5. Results (test split, mean ± sample std over seeds 0, 1, 2)

| Model | Val macro-F1 | Test macro-F1 | Test accuracy | Informative P | Informative R | Trainable params | Train time / run |
|---|---|---|---|---|---|---:|---:|
| Majority baseline | 0.402 | 0.402 | 0.671 | 0.671 | 1.000 | 0 | — |
| Text (TextCNN) | 0.783 ± 0.004 | 0.797 ± 0.007 | 0.824 ± 0.009 | 0.857 ± 0.010 | 0.885 ± 0.025 | 1,093,166 | 29 s |
| Text (BiLSTM) | 0.783 ± 0.014 | 0.801 ± 0.009 | 0.827 ± 0.008 | 0.860 ± 0.004 | 0.887 ± 0.010 | 1,275,266 | 35 s |
| Image | 0.787 ± 0.009 | 0.794 ± 0.008 | 0.824 ± 0.004 | 0.849 ± 0.014 | 0.898 ± 0.015 | 131,842 | 5 s |
| Fusion (TextCNN) | 0.845 ± 0.004 | 0.857 ± 0.007 | 0.874 ± 0.008 | 0.905 ± 0.008 | 0.908 ± 0.023 | 1,224,238 | 22 s |
| Fusion (BiLSTM) | 0.845 ± 0.012 | 0.857 ± 0.004 | 0.875 ± 0.002 | 0.898 ± 0.014 | 0.919 ± 0.018 | 1,406,338 | 33 s |

Trainable parameters exclude the frozen ResNet18 (its features are cached). Full tables, per-class
scores and the selected runs are in `results/summary/informative/final/`.

**Main observations**
- Fusion improves test macro-F1 by about 0.06 over either single modality, for every seed. Most of the gain
  is on `not_informative` (F1 0.72 → 0.81).
- Swapping TextCNN for BiLSTM inside the fusion model makes no measurable difference (0.857 vs 0.857), while
  BiLSTM costs about 50% more training time and energy.
- Text-only and image-only models perform similarly; the image model needs about 1/8 of the energy.
- Text models overfit quickly (training loss → ~0.05–0.1 while validation loss rises after epoch 2–4);
  early stopping keeps the best validation epoch.

**Published baseline (Ofli et al., 2020; accuracy only, different models — not our runs):** text CNN 80.8%,
image VGG16 83.3%, fusion 84.4%. Their text model uses pretrained word vectors and their image model is
VGG16, so the numbers are context rather than a like-for-like comparison
(`results/summary/informative/final/literature_comparison.md`).

## 6. Outputs

```
logs/
  run_log.csv                       the experiment log: one row per run (all types, incl. failed/interrupted)
  stage_summary.csv                 energy/emissions by workflow stage (report Table 1)
  model_summary.csv                 final models: F1 mean/std and energy per seed (report Table 2)
  runs/<id>/summary.json            one run's full record (config, hardware, energy per component, ranges, scores)
  runs/<id>/epochs.csv, tensorboard/, energy*.csv   per-epoch log, TensorBoard events, CodeCarbon samples
  runs/<id>/history.csv, metrics.json, predictions_valid.csv, train.log   our training outputs
  runs/<id>_test_eval…/predictions_test.csv, metrics.json                  linked test evaluation
results/
  runs/SHC-A-0xx/                   original Part A outputs (predictions, metrics, history); referenced by the
                                    converted records in logs/runs/
  summary/informative/final/        summary.md/.csv, runs.csv, selected_runs.json, literature_comparison.md,
                                    model_comparison.png, learning_curves_*.png, confusion_*.png,
                                    fusion_examples.png/.csv, fusion_case_counts.csv
  environment/partA/                energy_by_stage.png, energy_vs_performance.png, energy_by_component.png,
                                    environmental_indicators.csv
checkpoints/part_a_informative_fusion_textcnn_SHC-A-026.pt (+ .json with selection rule and SHA-256)
```

Part A experiment IDs are `SHC-A-<number>` (original scheme); later runs use the logger's
`<member>_<part>_<model>_seed<N>_<run type>_<timestamp>_<random>`. Each run is registered before it starts, so failed
and interrupted runs remain in the log. Read predictions with `tweet_id`/`image_id` as strings.

## 7. Selected checkpoint

`checkpoints/part_a_informative_fusion_textcnn_SHC-A-026.pt` — fusion (TextCNN), seed 2, epoch 4.
Fusion was the best model family on validation; fusion TextCNN and fusion BiLSTM tie on mean validation
macro-F1 (0.8453 vs 0.8455), so the cheaper TextCNN variant was chosen, and within it the seed with the median
validation macro-F1. Test scores were not used for this choice. The checkpoint stores the weights, vocabulary,
class order and configuration; step 6 above rebuilds and evaluates it.

## 8. Energy and emissions

Every run is tracked with CodeCarbon 3.3.2 through the group run logger (`scripts/run_logger.py`, settings in
`templates/measurement_method.json`): whole-machine mode, CPU energy **measured** through Intel RAPL, RAM
**estimated** (CodeCarbon heuristic), no GPU. Emissions use the group's central carbon intensity, **122 gCO₂/kWh**
(NESO Great Britain average, Nov 2025–Oct 2026), with a 94–237.6 g/kWh range; every record also carries a low/high
energy range. The image-feature extraction (2.23 Wh, 180 s) was **estimated**, because the RAPL counter was not
readable at the time. The idle laptop draws about 21 W (60 s idle-baseline record), roughly 41–46% of each learned
model's energy. Only aggregate energy, power, duration and hardware fields are kept; no location data.

| Workflow stage (Part A) | Runs | Runtime | Energy (central, range) |
|---|---:|---:|---:|
| Final training | 16 | 372 s | 4.66 Wh |
| Development (debug, exploratory, idle baseline, feature-extraction test) | 14 | 233 s | 2.28 Wh |
| Artefact production (feature extraction) | 1 | 180 s | 2.23 Wh |
| Evaluation (test) | 16 | 4 s | 0.06 Wh |
| **Total** | 47 | 790 s | **9.23 Wh** (6.37–15.01), **1.13 gCO₂e** |

Figures: `results/environment/partA/`. The original records used CodeCarbon's built-in UK value (237.6 g/kWh);
their emissions were recomputed at 122 g/kWh when converted (energy unchanged).

## 9. References and reused components

- Alam, F., Ofli, F. and Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. ICWSM.
- Ofli, F., Alam, F. and Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for
  Disaster Response. ISCRAM.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR — ResNet18 weights from torchvision.
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP — TextCNN design.
- CodeCarbon (https://github.com/mlco2/codecarbon) — energy and emissions tracking.
- Group run logger, aggregator and logging templates (`scripts/run_logger.py`, `scripts/aggregate_runs.py`,
  `scripts/energy_tensorboard.py`, `templates/`) — written by a group member (branch `Mehul`), used unchanged.
- NESO Carbon Intensity API (https://api.carbonintensity.org.uk/) — carbon-intensity figure.
