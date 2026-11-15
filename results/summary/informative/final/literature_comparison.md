# Comparison with Ofli et al. (2020) — accuracy only

Published values use pretrained word vectors (text CNN) and VGG16 (image); they are literature
values, not our runs. Our macro-F1 is reported separately in summary.md.

| Source | Model | Accuracy |
|---|---|---|
| Ofli et al. (2020) | Text CNN | 80.8% |
| Ofli et al. (2020) | Image VGG16 | 83.3% |
| Ofli et al. (2020) | Text + image fusion | 84.4% |
| Ours (test, mean ± std) | Majority baseline | 67.1% ± 0.0% |
| Ours (test, mean ± std) | Text (TextCNN) | 82.4% ± 0.9% |
| Ours (test, mean ± std) | Text (BiLSTM) | 82.7% ± 0.8% |
| Ours (test, mean ± std) | Image | 82.4% ± 0.4% |
| Ours (test, mean ± std) | Fusion (TextCNN) | 87.4% ± 0.8% |
| Ours (test, mean ± std) | Fusion (BiLSTM) | 87.5% ± 0.2% |
