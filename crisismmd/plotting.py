"""Consistent Matplotlib styling and the standard result figures (static PNGs for the report/video)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (colour-blind-safe adjacent pairs) and text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 9,
    })


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def learning_curves(histories, title, path):
    """histories: {seed: DataFrame with epoch, train_loss, val_loss, val_macro_f1}; best epoch marked."""
    apply_style()
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(10, 3.8))
    for index, (seed, history) in enumerate(sorted(histories.items())):
        first = index == 0
        loss_ax.plot(history["epoch"], history["train_loss"], color=SERIES[0], alpha=0.85,
                     label="Train loss" if first else None)
        loss_ax.plot(history["epoch"], history["val_loss"], color=SERIES[1], alpha=0.85, linestyle="--",
                     label="Validation loss" if first else None)
        best = history.loc[history["val_macro_f1"].idxmax()]
        f1_ax.plot(history["epoch"], history["val_macro_f1"], color=SERIES[index % len(SERIES)],
                   label=f"Seed {seed}")
        f1_ax.scatter([best["epoch"]], [best["val_macro_f1"]], s=48, color=SERIES[index % len(SERIES)],
                      edgecolor=SURFACE, linewidth=2, zorder=3)
    loss_ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    f1_ax.set(title="Validation macro-F1 (dot = selected epoch)", xlabel="Epoch", ylabel="Macro-F1")
    for ax in (loss_ax, f1_ax):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    save(fig, path)


def confusion_matrix(matrix, labels, title, path):
    """Counts with row percentages; rows = true class, columns = predicted class."""
    apply_style()
    matrix = np.asarray(matrix)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    size = 1.2 + 1.1 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color=SURFACE if shares[i, j] > 0.55 else TEXT)
    pretty = [label.replace("_", " ") for label in labels]
    ax.set_xticks(range(len(labels)), [_wrap(p, 14) for p in pretty], rotation=0)
    ax.set_yticks(range(len(labels)), [_wrap(p, 18) for p in pretty])
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(title, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, path)


def model_comparison(rows, metric_label, path, reference=None):
    """Dot plot: per-seed values (small dots) and mean ± std (large dot + bar) for each model config.

    rows: list of (label, [values]); reference: optional (label, value) drawn as a vertical line.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.6))
    every = np.concatenate([np.asarray(v, dtype=float) for _, v in rows])
    low, high = every.min(), every.max()
    pad = max(0.01, 0.08 * (high - low))
    for y, (label, values) in enumerate(rows):
        values = np.asarray(values, dtype=float)
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ax.plot([mean - std, mean + std], [y, y], color=SERIES[0], linewidth=6, alpha=0.25,
                solid_capstyle="round")
        ax.scatter(values, [y] * len(values), s=16, color=MUTED, zorder=3)
        ax.scatter([mean], [y], s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{mean:.3f} ± {std:.3f}", (max(values.max(), mean + std), y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(low - pad, high + 4 * pad)
    subtitle = "dot = mean, band = ± 1 std, grey = individual seeds"
    if reference:
        if low - pad <= reference[1] <= high + 4 * pad:
            ax.axvline(reference[1], color=MUTED, linestyle=":", linewidth=1.5)
        else:
            subtitle += f"; {reference[0]} = {reference[1]:.3f} (off scale)"
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric_label)
    ax.set_title(f"{metric_label} by model\n", fontsize=12)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    fig.tight_layout()
    save(fig, path)


def _wrap(text, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join(lines + [line])
