| Model | Seeds | Val macro-F1 | Valid macro-F1 | Valid accuracy | Valid weighted-F1 | Params | Epochs (best) | Train time (s) | Energy/run (kWh) |
|---|---|---|---|---|---|---|---|---|---|
| Text (TextCNN) | 1 | 0.530 ± 0.000 | 0.530 ± 0.000 | 0.727 ± 0.000 | 0.722 ± 0.000 | 922,033 | 10.0 (7.0) | 20.4 ± 0.0 | 3.15e-04 |
| Image | 1 | 0.591 ± 0.000 | 0.591 ± 0.000 | 0.766 ± 0.000 | 0.760 ± 0.000 | 132,613 | 7.0 (4.0) | 3.6 ± 0.0 | 3.41e-05 |
| Fusion (TextCNN) | 1 | 0.703 ± 0.000 | 0.703 ± 0.000 | 0.821 ± 0.000 | 0.819 ± 0.000 | 1,053,105 | 13.0 (10.0) | 31.0 ± 0.0 | 4.80e-04 |
| Fusion (TextCNN) + weighted loss | 1 | 0.692 ± 0.000 | 0.692 ± 0.000 | 0.811 ± 0.000 | 0.811 ± 0.000 | 1,053,105 | 12.0 (9.0) | 32.5 ± 0.0 | 4.11e-04 |
| Fusion (TextCNN) + weighted sampler | 1 | 0.678 ± 0.000 | 0.678 ± 0.000 | 0.821 ± 0.000 | 0.819 ± 0.000 | 1,053,105 | 11.0 (8.0) | 28.5 ± 0.0 | 3.50e-04 |

Per-class valid F1 (mean ± std) and number correct per seed:

| Model | affected_individuals | infrastructure_and_utility_damage | rescue_volunteering_or_donation_effort | other_relevant_information | not_humanitarian |
|---|---|---|---|---|---|
| Text (TextCNN) | 0.000 ± 0.000 (0/9) | 0.531 ± 0.000 (38/80) | 0.638 ± 0.000 (89/149) | 0.668 ± 0.000 (171/239) | 0.813 ± 0.000 (428/521) |
| Image | 0.000 ± 0.000 (0/9) | 0.720 ± 0.000 (63/80) | 0.688 ± 0.000 (95/149) | 0.740 ± 0.000 (159/239) | 0.808 ± 0.000 (447/521) |
| Fusion (TextCNN) | 0.308 ± 0.000 (2/9) | 0.786 ± 0.000 (66/80) | 0.770 ± 0.000 (112/149) | 0.796 ± 0.000 (182/239) | 0.857 ± 0.000 (457/521) |
| Fusion (TextCNN) + weighted loss | 0.286 ± 0.000 (2/9) | 0.768 ± 0.000 (58/80) | 0.752 ± 0.000 (120/149) | 0.812 ± 0.000 (207/239) | 0.842 ± 0.000 (422/521) |
| Fusion (TextCNN) + weighted sampler | 0.154 ± 0.000 (1/9) | 0.817 ± 0.000 (67/80) | 0.741 ± 0.000 (113/149) | 0.827 ± 0.000 (201/239) | 0.850 ± 0.000 (437/521) |

Mean ± sample standard deviation (ddof=1) over seeds. Confusion matrices use the run with the median validation macro-F1 (`selected_run` in summary.csv).
