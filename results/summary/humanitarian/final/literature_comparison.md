# Comparison with Ofli et al. (2020) — accuracy only

Published values use pretrained word vectors (text CNN) and VGG16 (image); they are literature
values, not our runs. Our macro-F1 is reported separately in summary.md.

| Source | Model | Accuracy |
|---|---|---|
| Ofli et al. (2020) | Text CNN | 70.4% |
| Ofli et al. (2020) | Image VGG16 | 76.8% |
| Ofli et al. (2020) | Text + image fusion | 78.4% |
| Ours (test, mean ± std) | Majority baseline | 52.8% ± 0.0% |
| Ours (test, mean ± std) | Text (TextCNN) | 71.9% ± 1.2% |
| Ours (test, mean ± std) | Image | 77.9% ± 0.4% |
| Ours (test, mean ± std) | Fusion (TextCNN) | 81.5% ± 0.9% |
| Ours (test, mean ± std) | Fusion (TextCNN) + weighted loss | 82.0% ± 1.4% |
