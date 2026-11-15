| Model | Seeds | Val macro-F1 | Valid macro-F1 | Valid accuracy | Valid weighted-F1 | Valid informative P | Valid informative R | Params | Epochs (best) | Train time (s) | Energy/run (kWh) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Text (TextCNN) | 1 | 0.783 ± 0.000 | 0.783 ± 0.000 | 0.814 ± 0.000 | 0.811 ± 0.000 | 0.845 ± 0.000 | 0.885 ± 0.000 | 1,093,166 | 7.0 (4.0) | 35.3 ± 0.0 | 3.15e-04 |
| Text (BiLSTM) | 1 | 0.780 ± 0.000 | 0.780 ± 0.000 | 0.809 ± 0.000 | 0.807 ± 0.000 | 0.848 ± 0.000 | 0.871 ± 0.000 | 1,275,266 | 7.0 (4.0) | 47.7 ± 0.0 | 5.12e-04 |
| Image | 1 | 0.786 ± 0.000 | 0.786 ± 0.000 | 0.820 ± 0.000 | 0.815 ± 0.000 | 0.838 ± 0.000 | 0.908 ± 0.000 | 131,842 | 6.0 (3.0) | 3.7 ± 0.0 | 4.03e-05 |
| Fusion (TextCNN) | 1 | 0.849 ± 0.000 | 0.849 ± 0.000 | 0.871 ± 0.000 | 0.869 ± 0.000 | 0.884 ± 0.000 | 0.930 ± 0.000 | 1,224,238 | 5.0 (2.0) | 19.7 ± 0.0 | 2.65e-04 |
| Fusion (BiLSTM) | 1 | 0.858 ± 0.000 | 0.858 ± 0.000 | 0.876 ± 0.000 | 0.875 ± 0.000 | 0.901 ± 0.000 | 0.916 ± 0.000 | 1,406,338 | 5.0 (2.0) | 32.1 ± 0.0 | 3.99e-04 |

Per-class valid F1 (mean ± std) and number correct per seed:

| Model | not_informative | informative |
|---|---|---|
| Text (TextCNN) | 0.702 ± 0.000 (345/517) | 0.865 ± 0.000 (935/1056) |
| Text (BiLSTM) | 0.700 ± 0.000 (352/517) | 0.859 ± 0.000 (920/1056) |
| Image | 0.701 ± 0.000 (331/517) | 0.871 ± 0.000 (959/1056) |
| Fusion (TextCNN) | 0.793 ± 0.000 (388/517) | 0.906 ± 0.000 (982/1056) |
| Fusion (BiLSTM) | 0.808 ± 0.000 (411/517) | 0.908 ± 0.000 (967/1056) |

Mean ± sample standard deviation (ddof=1) over seeds. Confusion matrices use the run with the median validation macro-F1 (`selected_run` in summary.csv).
