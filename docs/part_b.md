# Part B: Humanitarian categories (five classes) — reproduction guide and results

Predict one of five humanitarian categories for each tweet–image pair (CrisisMMD v2.0, official agreed-label
splits). Part B uses the same pipeline, settings and frozen image features as Part A (`docs/part_a.md`); this
document covers what differs, the exact commands and the results.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| not_humanitarian | 3,252 (53%) | 521 | 504 |
| other_relevant_information | 1,279 (21%) | 239 | 235 |
| rescue_volunteering_or_donation_effort | 912 (15%) | 149 | 126 |
| infrastructure_and_utility_damage | 612 (10%) | 80 | 81 |
| affected_individuals | 71 (1.2%) | 9 | 9 |

## 1. Exact commands

Run from the repository root. Steps 1–2 of `docs/part_a.md` (data and image features) must have been run; Part B
reuses the same feature cache (all 8,079 Part B images are among the 12,708 cached images).

```bash
export CRISISMMD_MEMBER=SHC

# 1. Exploratory runs (seed 0, validation only) used to choose the imbalance intervention
for m in "text --text-encoder textcnn" "image" "fusion --text-encoder textcnn" "fusion --text-encoder textcnn --imbalance weighted_loss" "fusion --text-encoder textcnn --imbalance weighted_sampler"; do
  python scripts/train.py --task humanitarian --model $m --run-type exploratory --purpose "Part B full-data check"
done

# 2. Final training (≈4 min on CPU): majority once, four learned configurations × seeds 0, 1, 2
python scripts/train.py --task humanitarian --model majority --run-type final --purpose "Part B final"
for seed in 0 1 2; do
  for m in "text --text-encoder textcnn" "image" "fusion --text-encoder textcnn" "fusion --text-encoder textcnn --imbalance weighted_loss"; do
    python scripts/train.py --task humanitarian --model $m --seed $seed --run-type final --purpose "Part B final"
  done
done

# 3. Timestamped median-validation selection, THEN linked test evaluation of every final run
python scripts/evaluate_final.py --task humanitarian

# 4. Run log and tables, results figures, energy figures
python -m scripts.aggregate_runs
python scripts/summarise.py --task humanitarian --run-type final
python scripts/energy_figures.py --part B

# 5. Re-evaluate the submitted checkpoint without retraining
python scripts/evaluate_checkpoint.py checkpoints/part_b_humanitarian_fusion_textcnn_weighted_loss_seed1.pt --split test
```

## 2. What differs from Part A

- **Text encoder: TextCNN only.** Part A showed TextCNN and BiLSTM tie on validation inside fusion while BiLSTM
  costs about 50% more energy, so Part B uses TextCNN (the brief requires text-only, image-only and fusion).
- **Vocabulary** rebuilt from the Part B training split only (5,389 tokens incl. `<pad>`, `<unk>`); each task is
  trained afresh.
- **Imbalance intervention: class-weighted cross-entropy**, applied to the fusion model with architecture,
  optimiser, batch size, epochs, early stopping and seeds unchanged. Weights come from the training labels only:
  `w_c = N / (C × n_c)` (N = 6,126 training rows, C = 5 classes, n_c = rows of class c):

  | Class | Weight |
  |---|---:|
  | affected_individuals | 17.256 |
  | infrastructure_and_utility_damage | 2.002 |
  | rescue_volunteering_or_donation_effort | 1.343 |
  | other_relevant_information | 0.958 |
  | not_humanitarian | 0.377 |

- **Why weighted loss rather than a weighted sampler:** in the seed-0 exploratory runs (validation only) weighted
  loss scored 0.692 macro-F1 and the sampler 0.678; weighted loss also keeps the data order identical to plain
  cross-entropy (a cleaner controlled comparison) and does not repeat the 71 rare examples ~17 times per epoch.
- **Test protocol:** final training never loads the test split. `scripts/evaluate_final.py` saved the median-validation
  selection (`results/selection/partB_humanitarian_median_validation_selection.json`, 2026-11-22 20:42:53 UTC,
  `test_results_inspected_before_selection: false`) and only then created one linked evaluation record per run.

## 3. Results (test split, mean ± sample std over seeds 0, 1, 2)

| Model | Val macro-F1 | Test macro-F1 | Test accuracy | Test weighted-F1 | Params | Best epoch (mean) | Train time |
|---|---|---|---|---|---:|---:|---:|
| Majority baseline | 0.137 | 0.138 | 0.528 | 0.365 | 0 | — | 2 s |
| Text (TextCNN) | 0.553 ± 0.021 | 0.521 ± 0.009 | 0.719 ± 0.012 | 0.714 ± 0.013 | 922,033 | 8.0 | 26 s |
| Image | 0.590 ± 0.002 | 0.596 ± 0.004 | 0.779 ± 0.004 | 0.771 ± 0.004 | 132,613 | 5.7 | 4 s |
| Fusion (TextCNN) | 0.667 ± 0.032 | 0.646 ± 0.024 | 0.815 ± 0.009 | 0.813 ± 0.010 | 1,053,105 | 7.3 | 26 s |
| **Fusion + weighted loss** | **0.704 ± 0.017** | **0.711 ± 0.033** | 0.820 ± 0.014 | 0.820 ± 0.014 | 1,053,105 | 9.0 | 32 s |

**Per-class test F1 (mean) and number correct per seed**

| Class (train n) | Text | Image | Fusion | Fusion + weighted loss |
|---|---|---|---|---|
| affected_individuals (71) | 0.000 (0, 0, 0 /9) | 0.000 (0, 0, 0 /9) | 0.056 (0, 0, 1 /9) | **0.348 (2, 2, 3 /9)** |
| infrastructure_and_utility_damage (612) | 0.479 | 0.781 | 0.786 | 0.799 |
| rescue_volunteering_or_donation_effort (912) | 0.651 | 0.592 | 0.717 | 0.733 |
| other_relevant_information (1,279) | 0.679 | 0.782 | 0.818 | 0.822 |
| not_humanitarian (3,252) | 0.796 | 0.823 | 0.853 | 0.853 |

**Main observations**
- **Fusion beats both single modalities** (+0.05 over image, +0.13 over text). Unlike Part A, image-only beats
  text-only, mainly on infrastructure damage (F1 0.78 vs 0.48), which is visible in photos.
- **Class-weighted loss improves macro-F1 by +0.065** at the same accuracy (0.820 vs 0.815). The gain is driven by
  the rarest class: `affected_individuals` goes from 1 correct out of 27 test predictions (3 seeds × 9) to 7 of 27.
- **Common vs rare trade-off:** with weighting, `not_humanitarian` recall falls (0.843 → 0.820) while its precision
  rises (0.865 → 0.890), so its F1 is unchanged; `other_relevant_information` gains recall (0.814 → 0.864) but loses
  precision (0.823 → 0.784). `affected_individuals` precision rises from 0.11 to 0.57 and recall from 0.04 to 0.26.
  Even with weighting, 4 of 9 `affected_individuals` test cases are still called `not_humanitarian` in the
  selected run, and 4 `not_humanitarian` posts are wrongly flagged as `affected_individuals`.
- **Noise:** with 9 `affected_individuals` examples in validation and test, one example changes macro-F1 by about
  0.02–0.03. Seed-to-seed spread is larger than in Part A (std up to 0.033), so differences below ~0.03 should
  not be over-interpreted. The weighted-loss gain is consistent on both validation and test.
- **Most frequent errors** (selected weighted run): `not_humanitarian` predicted as `other_relevant_information`
  (42) or `rescue_volunteering_or_donation_effort` (37), and the reverse confusions into `not_humanitarian`
  (27 and 22). Examples: `results/summary/humanitarian/final/error_examples.png`.

**Published baseline (Ofli et al., 2020; five-class accuracy only, different models — not our runs):** text CNN
70.4%, image VGG16 76.8%, fusion 78.4%. Ours: text 71.9%, image 77.9%, fusion 81.5% (82.0% with weighted loss).
Their text model uses pretrained word vectors and VGG16 image features, so this is context, not a like-for-like
comparison (`results/summary/humanitarian/final/literature_comparison.md`).

## 4. Outputs

```
logs/runs/SHC_B_*                              training records (+ linked *_test_eval records)
results/selection/partB_humanitarian_median_validation_selection.json
results/summary/humanitarian/final/            summary.md/.csv, runs.csv, selected_runs.json, literature_comparison.md,
                                               model_comparison.png, learning_curves_*.png, confusion_*.png,
                                               class_tradeoff.png/.csv, error_examples.png/.csv, confusion_pairs.csv
results/summary/humanitarian/exploratory/      the seed-0 runs used to choose the intervention
results/environment/partB/                     energy figures and environmental_indicators.csv
checkpoints/part_b_humanitarian_fusion_textcnn_weighted_loss_seed1.pt (+ .json)
```

## 5. Selected checkpoint

`checkpoints/part_b_humanitarian_fusion_textcnn_weighted_loss_seed1.pt` — fusion (TextCNN) with class-weighted
loss, seed 1, epoch 9. Fusion + weighted loss has the best mean validation macro-F1 (0.704 vs 0.667 for plain
cross-entropy); within it, seed 1 has the median validation macro-F1 (0.696). Test scores were not used for this
choice. Its own test result: macro-F1 0.682, accuracy 0.806, `affected_individuals` 2/9.

## 6. Energy and emissions (Part B)

Same measurement method as Part A (CodeCarbon via the group logger, CPU measured with RAPL, RAM estimated,
122 gCO₂/kWh). No feature extraction was needed: the Part A cache was reused.

| Workflow stage | Runs | Runtime | Energy |
|---|---:|---:|---:|
| Final training | 13 | 267 s | 3.37 Wh |
| Development (debug + exploratory) | 6 | 119 s | 1.61 Wh |
| Evaluation (test) | 13 | 4 s | 0.03 Wh |
| **Total** | 32 | 390 s | **5.01 Wh, 0.61 gCO₂e** |

Weighted loss costs about 15% more energy per seed than plain cross-entropy (0.40 vs 0.34 Wh; it trains ~2 epochs
longer) for +0.065 test macro-F1. Image-only remains by far the most energy-efficient model (≈13,500 macro-F1 per
kWh vs ≈1,800 for the fusion models). About 42–46% of each learned model's energy is the laptop's idle draw.

## 7. Hardware, software and references

Same as `docs/part_a.md` §4 and §9 (Intel Core i5-13500H CPU, 15.2 GiB RAM, no GPU; Python 3.12.3, torch 2.10.0+cpu,
CodeCarbon 3.3.2).
