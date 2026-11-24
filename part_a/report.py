"""Build every Part A table and figure from the run records.

    python -m part_a.report

Outputs (results/part_a/):
  results_per_seed.csv        model, seed, selected epoch, val/test metrics, params, time, energy
  results_summary.csv         mean and sample std over seeds 0/1/2 (+ majority baseline)
  published_comparison.csv    our test accuracy vs Ofli et al. (2020)
  median_seed_selection.json  confusion-matrix run per model, chosen by median VALIDATION F1
  fusion_examples.csv         six test examples where fusion helps / fails
  env_stage_summary.csv       Table 1: impact by workflow stage
  env_model_summary.csv       Table 2: performance vs environmental cost
  env_scaling_scenarios.csv   measured / 10x / 100x (fixed vs data-dependent)
  env_aggregate.json          run counts, epochs, runtime, wasted-compute ratio, uncertainty
  figures/*.png
Experiment log: logs/part_a/run_log.csv (rebuilt here too).
"""
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from . import config as C  # noqa: E402
from .energy import RunLogger, write_run_log  # noqa: E402

ORDER = ["majority", *C.MODELS]
NAMES = {"majority": "Majority class", "text_textcnn": "Text (TextCNN)",
         "text_bilstm": "Text (BiLSTM)", "image_mlp": "Image (ResNet18)",
         "fusion_textcnn": "Fusion (TextCNN)", "fusion_bilstm": "Fusion (BiLSTM)"}
COLORS = {"majority": "#9AA0A6", "text_textcnn": "#2F6DB5", "text_bilstm": "#7FA8D9",
          "image_mlp": "#E08A2C", "fusion_textcnn": "#1B9E77", "fusion_bilstm": "#7B5EA7"}
STAGES = ["development", "final_training", "evaluation", "part_c_extension",
          "artefact_production"]
STAGE_NAMES = {"development": "Development", "final_training": "Final training",
               "evaluation": "Evaluation", "part_c_extension": "Part C extension",
               "artefact_production": "Artefact production"}
PUBLISHED = {"text_textcnn": 80.8, "image_mlp": 83.3, "fusion_textcnn": 84.4}  # Ofli et al.
INK, MUTED, GRID = "#1F2933", "#616E7C", "#E4E7EB"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK, "axes.edgecolor": "#9AA5B1",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False, "figure.facecolor": "white"})


def save(fig, name, subtitle=None):
    if subtitle:
        fig.text(0.01, -0.02, subtitle, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Collect runs
# ---------------------------------------------------------------------------
def frame(runs):
    df = pd.DataFrame(runs)
    for col in ["energy_kwh", "energy_kwh_low", "energy_kwh_high", "emissions_kgco2e",
                "emissions_kgco2e_low", "emissions_kgco2e_high", "runtime_seconds",
                "loop_seconds", "setup_seconds", "epochs_completed", "engineering_estimate_kwh"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["run_status"] != "running"].copy()


def final_runs(df):
    """Latest completed final training run per (model, seed) and its linked test evaluation."""
    tr = df[(df.activity == "training") & (df.run_type == "final")
            & (df.dataset_fraction == 1.0)].sort_values("experiment_id")
    tr = tr.groupby(["model", "seed"], as_index=False).last()
    ev = df[(df.activity == "test_evaluation") & (df.run_type == "final")]
    ev = ev.sort_values("experiment_id").groupby("evaluated_experiment_id").last()
    rows = []
    for _, t in tr.iterrows():
        e = ev.loc[t.experiment_id] if t.experiment_id in ev.index else None
        vm, tm = t.metrics.get("valid", {}), (e.metrics.get("test", {}) if e is not None else {})
        rows.append({
            "model": t.model, "seed": int(t.seed), "train_id": t.experiment_id,
            "eval_id": e.experiment_id if e is not None else "",
            "selected_epoch": t.selected_epoch, "epochs_completed": t.epochs_completed,
            "run_status": t.run_status, "trainable_parameters": t.trainable_parameters,
            "val_macro_f1": vm.get("macro_f1"), "val_accuracy": vm.get("accuracy"),
            "test_macro_f1": tm.get("macro_f1"), "test_accuracy": tm.get("accuracy"),
            "test_informative_precision": tm.get("informative_precision"),
            "test_informative_recall": tm.get("informative_recall"),
            "test_not_informative_precision": tm.get("not_informative_precision"),
            "test_not_informative_recall": tm.get("not_informative_recall"),
            "test_confusion_matrix": tm.get("confusion_matrix"),
            "training_seconds": t.training_seconds if pd.notna(t.training_seconds)
            else t.loop_seconds,
            "train_runtime_seconds": t.runtime_seconds,
            "train_energy_kwh": t.energy_kwh, "train_emissions_kgco2e": t.emissions_kgco2e,
            "eval_energy_kwh": e.energy_kwh if e is not None else np.nan,
            "eval_emissions_kgco2e": e.emissions_kgco2e if e is not None else np.nan,
            "train_dir": (C.RUNS / t.experiment_id).as_posix(),
            "eval_dir": (C.RUNS / e.experiment_id).as_posix() if e is not None else ""})
    out = pd.DataFrame(rows)
    out["order"] = out.model.map(ORDER.index)
    return out.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)


def summarise(per_seed):
    metrics = ["val_macro_f1", "test_macro_f1", "test_accuracy", "test_informative_precision",
               "test_informative_recall", "training_seconds", "train_energy_kwh",
               "eval_energy_kwh"]
    rows = []
    for model, g in per_seed.groupby("model", sort=False):
        row = {"model": model, "name": NAMES.get(model, model), "seeds": len(g),
               "trainable_parameters": int(g.trainable_parameters.iloc[0]),
               "mean_selected_epoch": g.selected_epoch.mean()}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model figures
# ---------------------------------------------------------------------------
def fig_model_comparison(summary):
    cols = [("test_macro_f1", "Macro-F1"), ("test_accuracy", "Accuracy"),
            ("test_informative_precision", "Informative precision"),
            ("test_informative_recall", "Informative recall")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, cols):
        s = summary.set_index("model").reindex([m for m in ORDER if m in set(summary.model)])
        y = np.arange(len(s))[::-1]
        ax.barh(y, s[f"{col}_mean"], xerr=s[f"{col}_std"], color=[COLORS[m] for m in s.index],
                height=0.65, error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
        for yi, v in zip(y, s[f"{col}_mean"]):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [NAMES[m] for m in s.index])
        ax.set_xlim(0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Part A test performance by model", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", y=1.04)
    save(fig, "model_comparison.png",
         "Mean of seeds 0, 1, 2 (error bar = sample std). Majority baseline is deterministic.")


def fig_confusion(per_seed, selection):
    models = [m for m in ORDER if m in selection]
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow))
    for ax in np.ravel(axes)[len(models):]:
        ax.axis("off")
    for ax, m in zip(np.ravel(axes), models):
        row = per_seed[(per_seed.model == m) & (per_seed.seed == selection[m]["seed"])].iloc[0]
        cm = np.array(row.test_confusion_matrix)
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center",
                        fontsize=11, color="white" if pct[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["informative", "not inf."])
        ax.set_yticks([0, 1], ["informative", "not inf."])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"{NAMES[m]}  (seed {selection[m]['seed']})", fontsize=10.5)
    fig.suptitle("Test confusion matrices", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion_matrices_test.png",
         "Run per model chosen by median validation macro-F1 across seeds, before looking at "
         "test results. Cell = count and row-normalised share.")


def fig_learning_curves(per_seed):
    models = [m for m in C.MODELS if m in set(per_seed.model)]
    fig, axes = plt.subplots(2, len(models), figsize=(3.4 * len(models), 6.2), sharex=True)
    axes = np.atleast_2d(axes)
    for k, m in enumerate(models):
        for _, r in per_seed[per_seed.model == m].iterrows():
            h = pd.read_csv(f"{r.train_dir}/history.csv")
            axes[0, k].plot(h.epoch, h.train_loss, color=COLORS[m], lw=1.6, alpha=0.9)
            axes[0, k].plot(h.epoch, h.valid_loss, color=COLORS[m], lw=1.6, ls="--", alpha=0.9)
            axes[1, k].plot(h.epoch, h.valid_macro_f1, color=COLORS[m], lw=1.6, marker="o",
                            ms=3, alpha=0.9)
            axes[1, k].scatter([r.selected_epoch], [r.val_macro_f1], s=60, facecolor="white",
                               edgecolor=INK, zorder=5)
        axes[0, k].set_title(NAMES[m], fontsize=10.5)
        axes[1, k].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (solid train, dashed valid)")
    axes[1, 0].set_ylabel("Validation macro-F1")
    fig.suptitle("Learning curves, seeds 0/1/2", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "learning_curves.png", "Open circle = selected checkpoint (best validation "
         "macro-F1); training stops after 3 epochs without improvement.")


# ---------------------------------------------------------------------------
# Fusion examples
# ---------------------------------------------------------------------------
def fusion_examples(per_seed, selection, n_each=3):
    need = ["fusion_textcnn", "text_textcnn", "image_mlp"]
    if not all(m in selection for m in need):
        return None
    preds = {}
    for m in need + [x for x in ["fusion_bilstm"] if x in selection]:
        row = per_seed[(per_seed.model == m) & (per_seed.seed == selection[m]["seed"])].iloc[0]
        preds[m] = pd.read_csv(f"{row.eval_dir}/predictions_test.csv", dtype=str)
    base = preds["fusion_textcnn"][["tweet_id", "image_id", "event_name", "label", "image",
                                    "tweet_text"]].copy()
    for m, p in preds.items():
        base[f"{m}_pred"] = p["pred"].values
        base[f"{m}_p_inf"] = p["prob_informative"].astype(float).values
        base[f"{m}_ok"] = p["correct"].map({"True": True, "False": False}).values
    f, t, i = base.fusion_textcnn_ok, base.text_textcnn_ok, base.image_mlp_ok
    p_true = np.where(base.label == "informative", base.fusion_textcnn_p_inf,
                      1 - base.fusion_textcnn_p_inf)
    base["fusion_p_true_label"] = p_true
    groups = [("helps: fusion right, text and image both wrong", f & ~t & ~i, -p_true),
              ("helps: fusion right, one single modality wrong", f & (t ^ i), -p_true),
              ("fails: fusion wrong, text and image both right", ~f & t & i, p_true),
              ("fails: fusion wrong, one single modality right", ~f & (t ^ i), p_true)]
    counts = {name: int(mask.sum()) for name, mask, _ in groups}
    chosen = []
    for side in ("helps", "fails"):
        picked = []
        for name, mask, key in groups:
            if name.startswith(side) and len(picked) < n_each:
                cand = base[mask].assign(_k=key[mask]).sort_values(["_k", "tweet_id"])
                cand = cand.drop_duplicates("tweet_id")
                cand = cand[~cand.tweet_id.isin([c.tweet_id for c in picked])]
                for _, r in cand.head(n_each - len(picked)).iterrows():
                    r = r.copy()
                    r["category"] = name
                    picked.append(r)
        chosen += picked
    ex = pd.DataFrame(chosen).drop(columns="_k")
    ex.insert(0, "example", range(1, len(ex) + 1))
    ex.to_csv(C.OUT / "fusion_examples.csv", index=False, encoding="utf-8")
    (C.OUT / "fusion_example_counts.json").write_text(json.dumps(
        {"runs_compared": {m: selection[m] for m in preds}, "test_examples": len(base),
         "category_counts": counts}, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 13))
    for ax, (_, r) in zip(np.ravel(axes), ex.iterrows()):
        try:
            with Image.open(C.DATA / r.image) as im:
                im = im.convert("RGB")
                im.thumbnail((420, 300))
                ax.imshow(im)
        except OSError:
            ax.text(0.5, 0.5, "image unavailable", ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        helps = r.category.startswith("helps")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#1B9E77" if helps else "#C0392B")
            s.set_linewidth(3)
        ax.set_title(f"#{r.example}  {'FUSION HELPS' if helps else 'FUSION FAILS'}  |  "
                     f"true: {r.label}", fontsize=10,
                     color="#1B9E77" if helps else "#C0392B")
        tweet = textwrap.fill(" ".join(str(r.tweet_text).split())[:230], 62)
        marks = "  |  ".join(f"{lab}: {'right' if r[f'{m}_ok'] else 'WRONG'} {r[f'{m}_p_inf']:.2f}"
                           for m, lab in [("text_textcnn", "Text"), ("image_mlp", "Image"),
                                          ("fusion_textcnn", "Fusion")])
        ax.set_xlabel(f"{tweet}\n\nP(informative)  {marks}", fontsize=8.5, color=INK,
                      loc="left")
    for ax in np.ravel(axes)[len(ex):]:
        ax.axis("off")
    fig.suptitle("Where simple fusion helps or fails (test set)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.08, hspace=0.62,
                        wspace=0.08)
    save(fig, "fusion_examples.png", "Median-validation-seed runs of Text (TextCNN), Image "
         "(ResNet18) and Fusion (TextCNN). right/WRONG = prediction correctness, number = "
         "P(informative) from that model.")
    return ex


# ---------------------------------------------------------------------------
# Environmental analysis
# ---------------------------------------------------------------------------
def stage_table(df):
    total = df.energy_kwh.sum()
    rows = []
    for st in STAGES:
        g = df[df.workflow_stage == st]
        rows.append({"stage": STAGE_NAMES[st], "runs": len(g),
                     "runtime_s": g.runtime_seconds.sum(), "energy_kwh": g.energy_kwh.sum(),
                     "energy_kwh_low": g.energy_kwh_low.sum(),
                     "energy_kwh_high": g.energy_kwh_high.sum(),
                     "emissions_kgco2e": g.emissions_kgco2e.sum(),
                     "emissions_kgco2e_low": g.emissions_kgco2e_low.sum(),
                     "emissions_kgco2e_high": g.emissions_kgco2e_high.sum(),
                     "share": g.energy_kwh.sum() / total if total else 0,
                     "notes": _stage_note(st, g)})
    t = pd.DataFrame(rows)
    tot = t.drop(columns=["stage", "notes"]).sum().to_dict()
    t.loc[len(t)] = {"stage": "Total", **tot, "notes": ""}
    return t


def _stage_note(stage, g):
    if g.empty:
        return "No runs" + (" (Part C not attempted in this log)" if stage == "part_c_extension"
                            else "")
    parts = [f"{k}: {v}" for k, v in g.activity.value_counts().items()]
    failed = (g.run_type == "failed").sum()
    return "; ".join(parts) + (f"; {failed} failed" if failed else "")


def model_table(per_seed, summary):
    rows = []
    s = summary.set_index("model")
    maj = s.loc["majority"] if "majority" in s.index else None
    for model, g in per_seed.groupby("model", sort=False):
        e_seed = (g.train_energy_kwh + g.eval_energy_kwh.fillna(0)).mean()
        c_seed = (g.train_emissions_kgco2e + g.eval_emissions_kgco2e.fillna(0)).mean()
        f1 = s.loc[model, "test_macro_f1_mean"]
        row = {"model": model, "name": NAMES[model], "seeds": len(g),
               "test_macro_f1_mean": f1, "test_macro_f1_std": s.loc[model, "test_macro_f1_std"],
               "energy_kwh_all_seeds": (g.train_energy_kwh + g.eval_energy_kwh.fillna(0)).sum(),
               "emissions_kgco2e_all_seeds":
                   (g.train_emissions_kgco2e + g.eval_emissions_kgco2e.fillna(0)).sum(),
               "energy_kwh_per_seed": e_seed, "emissions_kgco2e_per_seed": c_seed,
               "f1_per_kwh": f1 / e_seed if e_seed else np.nan,
               "f1_per_kgco2e": f1 / c_seed if c_seed else np.nan}
        if maj is not None and model != "majority":
            c_maj = per_seed[per_seed.model == "majority"]
            c_maj = (c_maj.train_emissions_kgco2e + c_maj.eval_emissions_kgco2e.fillna(0)).mean()
            gain = (f1 - maj.test_macro_f1_mean) * 100
            row["marginal_gco2e_per_f1_point_vs_majority"] = (
                (c_seed - c_maj) * 1000 / gain if gain > 0 else np.nan)
        rows.append(row)
    t = pd.DataFrame(rows)
    learned = t[t.model != "majority"]
    notes = []
    for _, r in t.iterrows():
        if r.model == "majority":
            notes.append("Reference: no learning, near-zero energy, lowest F1")
            continue
        dom = learned[(learned.model != r.model)
                      & (learned.test_macro_f1_mean >= r.test_macro_f1_mean)
                      & (learned.energy_kwh_per_seed < r.energy_kwh_per_seed)]
        notes.append("Environmentally dominated by " + ", ".join(dom.name) if len(dom)
                     else "Not dominated (no cheaper model scores higher)")
    t["interpretation"] = notes
    return t


def scaling(df):
    """Split measured energy into fixed (setup) and data-dependent (loop) parts, then scale.

    Per run: data-dependent share = loop_seconds / runtime_seconds (time inside the
    epoch / inference loop); the rest (tracker start, data and model loading, saving) is
    treated as fixed per run. Runs without a loop (idle baseline, report generation) are
    fixed. Assumes roughly constant power within a run.
    """
    d = df.copy()
    frac = (d.loop_seconds / d.runtime_seconds).clip(0, 1).fillna(0)
    frac[d.activity.isin(["idle_baseline", "report_generation"])] = 0
    d["e_data"] = d.energy_kwh.fillna(0) * frac
    d["e_fixed"] = d.energy_kwh.fillna(0) - d.e_data
    by = d.groupby("workflow_stage")[["e_fixed", "e_data"]].sum()
    m = C.MEASUREMENT_METHOD
    lo_f, hi_f = m["energy_factor_low"], m["energy_factor_high"] * m["unmeasured_overhead_high"]
    ci, (ci_lo, ci_hi) = m["carbon_intensity_gco2e_per_kwh"], m["carbon_intensity_range_gco2e_per_kwh"]
    n_runs = len(d)
    fixed, data_dep = by.e_fixed.sum(), by.e_data.sum()
    rows = []
    for name, s, runs, note in [
            ("Measured", 1, "Actual", "Measured CodeCarbon totals for every logged run."),
            ("Medium scale", 10, "Same", "N x10; same epochs H, runs R, seeds, hardware. "
             "Data-dependent energy x10 (feature extraction, tokenisation, training "
             "batches, prediction); fixed setup per run unchanged."),
            ("Large scale", 100, "Same", "N x100; as above. Feature extraction alone becomes "
             "hours of CPU time; a GPU would change the hardware-efficiency factor.")]:
        e = fixed + s * data_dep
        rows.append({"scenario": name, "dataset": f"{s}x", "runs": runs if s == 1 else
                     f"{n_runs} (same)", "fixed_kwh": fixed, "data_dependent_kwh": s * data_dep,
                     "energy_kwh": e, "energy_kwh_low": e * lo_f, "energy_kwh_high": e * hi_f,
                     "emissions_kgco2e": e * ci / 1000,
                     "emissions_kgco2e_low": e * lo_f * ci_lo / 1000,
                     "emissions_kgco2e_high": e * hi_f * ci_hi / 1000, "assumptions": note})
    # Hyperparameter search sensitivity: E_search ~ N x H x S x K
    ft = by.loc["final_training"] if "final_training" in by.index else pd.Series(
        {"e_fixed": 0, "e_data": 0})
    K = 5
    e_search = fixed + 100 * data_dep + (K - 1) * (ft.e_fixed + 100 * ft.e_data)
    rows.append({"scenario": "Large scale + search", "dataset": "100x",
                 "runs": f"final training x{K} configurations", "fixed_kwh": np.nan,
                 "data_dependent_kwh": np.nan, "energy_kwh": e_search,
                 "energy_kwh_low": e_search * lo_f, "energy_kwh_high": e_search * hi_f,
                 "emissions_kgco2e": e_search * ci / 1000,
                 "emissions_kgco2e_low": e_search * lo_f * ci_lo / 1000,
                 "emissions_kgco2e_high": e_search * hi_f * ci_hi / 1000,
                 "assumptions": f"Sensitivity: E_search ~ N x H x S x K with K={K} "
                 "configurations x 3 seeds at 100x data; multiplicative growth."})
    return pd.DataFrame(rows), by


def storage_scaling():
    sizes = {"feature_cache_bytes": C.FEATURE_CACHE.stat().st_size if C.FEATURE_CACHE.exists()
             else 0,
             "predictions_bytes": sum(p.stat().st_size for p in C.RUNS.glob("*/predictions_*.csv")),
             "checkpoints_bytes": sum(p.stat().st_size for p in C.RUNS.glob("*/best.pt"))}
    return {k: {"1x_MB": v / 1e6, "10x_MB": v * (10 if k != "checkpoints_bytes" else 1) / 1e6,
                "100x_MB": v * (100 if k != "checkpoints_bytes" else 1) / 1e6}
            for k, v in sizes.items()} | {
        "note": "Cache and predictions scale with examples. Checkpoints are listed as fixed, "
                "but the embedding table grows sub-linearly with vocabulary size."}


def aggregate(df, per_seed):
    training = df[df.activity == "training"]
    failed = df[df.run_type == "failed"]
    e_train = training.energy_kwh.sum()
    m = C.MEASUREMENT_METHOD
    return {
        "runs_total": len(df),
        "runs_by_type": df.run_type.value_counts().to_dict(),
        "runs_by_status": df.run_status.value_counts().to_dict(),
        "runs_by_activity": df.activity.value_counts().to_dict(),
        "epochs_completed_total": float(training.epochs_completed.fillna(0).sum()),
        "runtime_seconds_total": float(df.runtime_seconds.sum()),
        "energy_kwh_total": float(df.energy_kwh.sum()),
        "energy_kwh_range": [float(df.energy_kwh_low.sum()), float(df.energy_kwh_high.sum())],
        "emissions_kgco2e_total": float(df.emissions_kgco2e.sum()),
        "emissions_kgco2e_range": [float(df.emissions_kgco2e_low.sum()),
                                   float(df.emissions_kgco2e_high.sum())],
        "engineering_estimate_kwh_total": float(df.engineering_estimate_kwh.sum()),
        "wasted_compute_ratio": float(failed.energy_kwh.sum() / e_train) if e_train else None,
        "wasted_compute_definition": "(E_failed + E_unreported_exploratory) / E_total_training; "
                                     "no unreported exploratory runs (all runs are logged).",
        "runs_missing_energy": df[df.energy_kwh.isna()].experiment_id.tolist(),
        "superseded_final_runs": sorted(
            set(df[(df.activity == "training") & (df.run_type == "final")].experiment_id)
            - set(per_seed.train_id)),
        "measurement_method": m,
        "f1_scale": "0-1",
        "std": "sample standard deviation (ddof=1) over seeds 0, 1, 2",
    }


def fig_energy_by_stage(stage):
    t = stage[stage.stage != "Total"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2F6DB5", "#1B9E77", "#E08A2C", "#9AA0A6", "#7B5EA7"][::-1]
    err = [t.energy_kwh - t.energy_kwh_low, t.energy_kwh_high - t.energy_kwh]
    ax.barh(t.stage, t.energy_kwh * 1000, xerr=np.array(err) * 1000, color=colors,
            height=0.6, error_kw={"ecolor": MUTED, "capsize": 3, "lw": 1})
    for y, (_, r) in enumerate(t.iterrows()):
        ax.text(r.energy_kwh_high * 1000 * 1.02 + 0.2, y,
                f"{r.energy_kwh * 1000:.1f} Wh  ·  {r.share:.0%}  ·  "
                f"{r.emissions_kgco2e * 1000:.2f} gCO₂e  ·  {int(r.runs)} runs",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("Energy (Wh)")
    ax.set_xlim(0, max(t.energy_kwh_high.max() * 1000 * 1.9, 1))
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the energy went: Part A by workflow stage")
    save(fig, "energy_by_stage.png", "Bar = CodeCarbon central estimate; whisker = "
         "uncertainty range (0.5x to 2.5x, CPU power estimated). Emissions at 122 gCO₂e/kWh.")


def fig_energy_by_model(mt, per_seed):
    t = mt.set_index("model").reindex([m for m in ORDER if m in set(mt.model)])
    g = per_seed.groupby("model")[["train_energy_kwh", "eval_energy_kwh"]].sum().reindex(t.index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(t))[::-1]
    ax.barh(y, g.train_energy_kwh * 1000, color=[COLORS[m] for m in t.index], height=0.6,
            label="Training (all seeds)")
    ax.barh(y, g.eval_energy_kwh.fillna(0) * 1000, left=g.train_energy_kwh * 1000,
            color=[COLORS[m] for m in t.index], alpha=0.45, height=0.6, hatch="///",
            edgecolor="white", label="Test evaluation")
    for yi, (m, r) in zip(y, t.iterrows()):
        ax.text(r.energy_kwh_all_seeds * 1000 * 1.02 + 0.05, yi,
                f"{r.energy_kwh_all_seeds * 1000:.2f} Wh  ·  "
                f"{r.emissions_kgco2e_all_seeds * 1000:.2f} gCO₂e", va="center", fontsize=9)
    ax.set_yticks(y, [NAMES[m] for m in t.index])
    ax.set_xlim(0, t.energy_kwh_all_seeds.max() * 1000 * 1.45)
    ax.set_xlabel("Energy (Wh)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Energy and emissions by model family (final runs)")
    save(fig, "energy_by_model.png", "Sum over seeds 0, 1, 2 (majority: one deterministic "
         "run). Hatched = separate test-evaluation records.")


def fig_energy_vs_performance(per_seed, mt):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in [x for x in ORDER if x in set(per_seed.model)]:
        g = per_seed[per_seed.model == m]
        e = (g.train_energy_kwh + g.eval_energy_kwh.fillna(0)) * 1000
        ax.scatter(e, g.test_macro_f1, color=COLORS[m], alpha=0.35, s=30)
        r = mt[mt.model == m].iloc[0]
        ax.errorbar(r.energy_kwh_per_seed * 1000, r.test_macro_f1_mean,
                    yerr=r.test_macro_f1_std, xerr=e.std(ddof=1) if len(e) > 1 else 0,
                    fmt="o", ms=10, color=COLORS[m], mec="white", mew=1.5, capsize=3,
                    label=f"{NAMES[m]}  ({r.f1_per_kwh:,.0f} F1/kWh)")
    ax.set_xscale("log")
    ax.set_xlabel("Energy per run: training + test evaluation (Wh, log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Performance vs energy: is the extra computation worth it?")
    save(fig, "energy_vs_performance.png", "Large marker = mean over seeds (error bars = "
         "std); small dots = individual seeds. Up and to the left is better.")


def fig_scaling(sc):
    t = sc.iloc[:3]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(t))
    a1.bar(x, t.energy_kwh, color=["#2F6DB5", "#1B9E77", "#7B5EA7"], width=0.55,
           yerr=[t.energy_kwh - t.energy_kwh_low, t.energy_kwh_high - t.energy_kwh],
           error_kw={"ecolor": MUTED, "capsize": 4})
    a1.set_yscale("log")
    for xi, (_, r) in zip(x, t.iterrows()):
        a1.text(xi, r.energy_kwh_high * 1.25, f"{r.energy_kwh:.3g} kWh\n"
                f"{r.emissions_kgco2e * 1000:.3g} gCO₂e", ha="center", fontsize=9)
    a1.set_xticks(x, [f"{d}\n{s}" for d, s in zip(t.dataset, t.scenario)])
    a1.set_ylabel("Energy (kWh, log scale)")
    a1.set_ylim(t.energy_kwh_low.min() / 2, t.energy_kwh_high.max() * 8)
    a1.set_title("Projected energy for larger datasets")
    share_f = t.fixed_kwh / t.energy_kwh
    a2.bar(x, share_f * 100, color="#9AA0A6", width=0.55, label="Fixed (per-run setup)")
    a2.bar(x, (1 - share_f) * 100, bottom=share_f * 100, color="#1B9E77", width=0.55,
           label="Data-dependent (scales with N)")
    for xi, v in zip(x, share_f):
        a2.text(xi, 101, f"{v:.0%} fixed", ha="center", fontsize=9)
    a2.set_xticks(x, t.dataset)
    a2.set_ylim(0, 110)
    a2.set_ylabel("Share of energy (%)")
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    a2.set_title("Fixed costs shrink as data grows")
    fig.tight_layout()
    save(fig, "energy_scaling.png", "Same runs, epochs, seeds and hardware; data-dependent "
         "energy scaled linearly with N. Whiskers = uncertainty range.")


def fig_estimate_check(df):
    d = df.dropna(subset=["energy_kwh", "engineering_estimate_kwh"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for act, col in [("training", "#1B9E77"), ("test_evaluation", "#E08A2C"),
                     ("feature_extraction", "#2F6DB5"), ("idle_baseline", "#9AA0A6"),
                     ("report_generation", "#7B5EA7")]:
        g = d[d.activity == act]
        ax.scatter(g.energy_kwh * 1000, g.engineering_estimate_kwh * 1000, s=28, color=col,
                   label=act.replace("_", " "), alpha=0.85)
    lim = [d[["energy_kwh", "engineering_estimate_kwh"]].min().min() * 1000 * 0.8,
           d[["energy_kwh", "engineering_estimate_kwh"]].max().max() * 1000 * 1.2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CodeCarbon (Wh)")
    ax.set_ylabel("Utilisation-based estimate, guidance 5.2 (Wh)")
    ax.legend(fontsize=8)
    ax.set_title("Two energy estimates per run")
    save(fig, "energy_estimate_check.png", "CodeCarbon assumes a constant 42.5 W CPU on this "
         "laptop; the 5.2 estimate uses measured CPU utilisation x 45 W base power.")


# ---------------------------------------------------------------------------
def main():
    style()
    with RunLogger(run_type="final", workflow_stage="artefact_production",
                   activity="report_generation", model="n/a",
                   measurement_boundary="Reading run records and predictions, computing "
                   "tables and rendering Matplotlib figures.") as run:
        run.update(trainable_parameters=0, cache_used=True,
                   reuse_notes="Uses saved predictions/checkpoint metadata; no retraining.")
        df = frame(write_run_log())
        df = df[df.experiment_id != run.id]
        per_seed = final_runs(df)
        if per_seed.empty:
            raise SystemExit("No final training runs yet.")
        summary = summarise(per_seed)
        per_seed.drop(columns=["train_dir", "eval_dir"]).to_csv(
            C.OUT / "results_per_seed.csv", index=False)
        summary.to_csv(C.OUT / "results_summary.csv", index=False)

        selection = {}
        for m, g in per_seed.groupby("model"):
            g = g.sort_values(["val_macro_f1", "seed"]).reset_index(drop=True)
            r = g.iloc[(len(g) - 1) // 2]
            selection[m] = {"seed": int(r.seed), "train_id": r.train_id, "eval_id": r.eval_id,
                            "val_macro_f1": r.val_macro_f1}
        (C.OUT / "median_seed_selection.json").write_text(json.dumps(selection, indent=2))

        pub = [{"model": NAMES[m], "ours_test_accuracy_mean_%":
                round(summary.set_index("model").loc[m, "test_accuracy_mean"] * 100, 1),
                "ours_std": round(summary.set_index("model").loc[m, "test_accuracy_std"] * 100, 1),
                "ofli_2020_accuracy_%": v,
                "their_setup": {"text_textcnn": "CNN + pretrained word2vec",
                                "image_mlp": "VGG16 (fine-tuned)",
                                "fusion_textcnn": "VGG16 + text CNN"}[m]}
               for m, v in PUBLISHED.items() if m in set(summary.model)]
        pd.DataFrame(pub).to_csv(C.OUT / "published_comparison.csv", index=False)

        fig_model_comparison(summary)
        fig_confusion(per_seed, selection)
        fig_learning_curves(per_seed[per_seed.model != "majority"])
        fusion_examples(per_seed, selection)

        stage = stage_table(df)
        stage.to_csv(C.OUT / "env_stage_summary.csv", index=False)
        mt = model_table(per_seed, summary)
        mt.to_csv(C.OUT / "env_model_summary.csv", index=False)
        sc, _ = scaling(df)
        sc.to_csv(C.OUT / "env_scaling_scenarios.csv", index=False)
        agg = aggregate(df, per_seed)
        agg["storage_scaling"] = storage_scaling()
        agg["note"] = (f"Tables exclude this report-generation run ({run.id}); it appears in "
                       "the run log and in the next regeneration.")
        (C.OUT / "env_aggregate.json").write_text(json.dumps(agg, indent=2, default=float))

        fig_energy_by_stage(stage)
        fig_energy_by_model(mt, per_seed)
        fig_energy_vs_performance(per_seed[per_seed.model != "majority"],
                                  mt[mt.model != "majority"])
        fig_scaling(sc)
        fig_estimate_check(df)
        run.mark_training_start()  # nothing data-dependent; keeps setup/loop columns filled
        run.mark_training_end()

    pd.set_option("display.width", 200)
    print(summary[["name", "seeds", "test_macro_f1_mean", "test_macro_f1_std",
                   "test_accuracy_mean", "test_informative_precision_mean",
                   "test_informative_recall_mean", "trainable_parameters",
                   "training_seconds_mean"]].round(4).to_string(index=False))
    print("\n" + stage[["stage", "runs", "runtime_s", "energy_kwh", "emissions_kgco2e",
                        "share"]].round(6).to_string(index=False))
    print(f"\nOutputs in {C.OUT}")


if __name__ == "__main__":
    main()
