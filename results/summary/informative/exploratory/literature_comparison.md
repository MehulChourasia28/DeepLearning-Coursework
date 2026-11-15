# Comparison with Ofli et al. (2020) — accuracy only

Published values use pretrained word vectors (text CNN) and VGG16 (image); they are literature
values, not our runs. Our macro-F1 is reported separately in summary.md.

| Source | Model | Accuracy |
|---|---|---|
| Ofli et al. (2020) | Text CNN | 80.8% |
| Ofli et al. (2020) | Image VGG16 | 83.3% |
| Ofli et al. (2020) | Text + image fusion | 84.4% |
| Ours (valid, mean ± std) | Text (TextCNN) | 81.4% ± 0.0% |
| Ours (valid, mean ± std) | Text (BiLSTM) | 80.9% ± 0.0% |
| Ours (valid, mean ± std) | Image | 82.0% ± 0.0% |
| Ours (valid, mean ± std) | Fusion (TextCNN) | 87.1% ± 0.0% |
| Ours (valid, mean ± std) | Fusion (BiLSTM) | 87.6% ± 0.0% |
