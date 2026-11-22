# Comparison with Ofli et al. (2020) — accuracy only

Published values use pretrained word vectors (text CNN) and VGG16 (image); they are literature
values, not our runs. Our macro-F1 is reported separately in summary.md.

| Source | Model | Accuracy |
|---|---|---|
| Ofli et al. (2020) | Text CNN | 70.4% |
| Ofli et al. (2020) | Image VGG16 | 76.8% |
| Ofli et al. (2020) | Text + image fusion | 78.4% |
| Ours (valid, mean ± std) | Text (TextCNN) | 72.7% ± 0.0% |
| Ours (valid, mean ± std) | Image | 76.6% ± 0.0% |
| Ours (valid, mean ± std) | Fusion (TextCNN) | 82.1% ± 0.0% |
| Ours (valid, mean ± std) | Fusion (TextCNN) + weighted loss | 81.1% ± 0.0% |
| Ours (valid, mean ± std) | Fusion (TextCNN) + weighted sampler | 82.1% ± 0.0% |
