# Part A: binary informativeness, with environmental logging

Predict `informative` vs `not_informative` on CrisisMMD v2.0 (official agreed-label splits:
9,601 / 1,573 / 1,534).

## Models

| Model | Text branch | Image branch | Head |
|---|---|---|---|
| Majority class | none | none | predicts the training majority (`informative`) |
| Text (TextCNN) | 128-d embeddings, kernels 2/3/4/5 x 64 filters, masked max-pool, 256-d output | none | MLP 256→256→2 |
| Text (BiLSTM) | 128-d embeddings, one-layer BiLSTM (128 per direction), final states, 256-d output | none | MLP 256→256→2 |
| Image | none | frozen ResNet18 (ImageNet), cached 512-d features | MLP 512→256→2 |
| Fusion (TextCNN) | TextCNN, 256-d | frozen 512-d | concatenate (768) → MLP 768→256→2 |
| Fusion (BiLSTM) | **BiLSTM**, 256-d | **same** | **same head** |

Both text encoders output 256-d, so the TextCNN→BiLSTM swap leaves the image branch and the
classifier unchanged. Every MLP head uses dropout 0.5.

Settings from the brief: training-only vocabulary (min frequency 2), 64-token limit, Adam
(lr 1e-3), batch size 64, at most 20 epochs, early stopping with patience 3 on validation
macro-F1. Seeds are 0, 1 and 2; the majority baseline is one deterministic run. The
confusion-matrix run for each model is the seed with the median validation macro-F1, chosen
before looking at test results.

## Commands (CPU)

```powershell
.venv\Scripts\Activate.ps1
python -m part_a.run_all                 # everything, in order (about 15-30 min on CPU)
```

Individual steps:

```powershell
python -m part_a.idle_baseline --seconds 60
python -m part_a.extract_features                    # writes data/crisismmd/features/resnet18_imagenet1k_v1.npz
python -m part_a.train --model fusion_textcnn --seeds 0 --run-type debug --fraction 0.05 --max-epochs 1
python -m part_a.train --model majority --seeds 0 --run-type final --test
python -m part_a.train --model fusion_bilstm --seeds 0 1 2 --run-type final --test
python -m part_a.report                              # tables + figures
```

If you repeat a run, pass `--rerun-of AYN-A-0xx --rerun-reason "..."`. The report always
uses the latest final run per model and seed, and lists superseded runs in `env_aggregate.json`.

## Environmental logging

`part_a/energy.py` creates one record per run, following the environmental-impact guidance:

- **Registration.** `results/part_a/runs/AYN-A-###/run.json` is written with status
  `running` before measurement starts.
- **Classification.** Each run has a run type (debug, exploratory, final, failed), a workflow
  stage (development, final_training, evaluation, part_c_extension, artefact_production) and
  an activity (training, test_evaluation, feature_extraction, idle_baseline,
  report_generation).
- **Measurement.** Energy is tracked with CodeCarbon `OfflineEmissionsTracker` in machine mode
  (GBR), started and stopped around each run. On this Windows laptop CodeCarbon cannot read a
  CPU power counter, so it assumes a constant 42.5 W (50% of a generic 85 W TDP). Each record
  therefore also stores a guidance §5.2 engineering estimate based on sampled CPU utilisation
  × 45 W base power plus RAM.
- **Emissions.** Calculated at 122 gCO₂e/kWh: the NESO 12-month GB average, the same value as
  the group's method. The range is 94–237.6 g/kWh.
- **Uncertainty.** Energy is bracketed at 0.5× to 2.5× (CPU power estimated, plus unmetered
  overhead).
- **Failures.** Failed and interrupted runs are kept: `run_type` becomes `failed` and
  `original_run_type` keeps the original purpose.
- **Fixed vs data-dependent time.** `setup_seconds` (fixed) and `loop_seconds`
  (data-dependent) are recorded separately for the scaling analysis.

The submitted experiment log is `logs/part_a/run_log.csv`, with full records in `run_log.json`.

## Outputs (`results/part_a/`)

| File | Contents |
|---|---|
| `results_per_seed.csv`, `results_summary.csv` | macro-F1, accuracy, informative precision/recall, parameters, training time (mean ± sample std) |
| `published_comparison.csv` | test accuracy compared with Ofli et al. (2020) |
| `median_seed_selection.json` | the run chosen for each confusion matrix |
| `fusion_examples.csv`, `fusion_example_counts.json` | 3 test examples where fusion helps and 3 where it fails |
| `env_stage_summary.csv` | Table 1: impact by workflow stage |
| `env_model_summary.csv` | Table 2: macro-F1, energy, emissions, F1/kWh, F1/kgCO₂e, marginal carbon cost, dominance |
| `env_scaling_scenarios.csv` | 1×, 10× and 100× scenarios plus a hyperparameter-search sensitivity case |
| `env_aggregate.json` | run counts, epochs, runtime, wasted-compute ratio, storage scaling |
| `figures/` | model comparison, confusion matrices, learning curves, fusion examples, energy by stage, energy by model, energy vs performance, scaling, estimate cross-check |

Exclusions (state these in the report): the monitor, network, lighting, embodied carbon,
dataset download and report writing. Charger losses appear only in the upper bound.
