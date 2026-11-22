#!/usr/bin/env python3
"""Environmental-impact figures and indicators for one coursework part, from logs/run_log.csv.

Same layout as the group's Part A notebook figures:
  energy_by_stage.png        energy by workflow stage, whiskers = low/high uncertainty range
  energy_vs_performance.png  energy per trained model (training + linked test evaluation, per seed)
                             and test macro-F1 against energy
  energy_by_component.png    CPU / GPU / RAM energy per seed, with the idle machine's draw over the same
                             measured time (from the latest idle-baseline record on the same CPU)
  environmental_indicators.csv  macro-F1 per kWh and per kgCO2e, marginal emissions vs majority, idle share

Run after `python -m scripts.aggregate_runs`. Output: results/environment/part<P>/

Example:
  python scripts/energy_figures.py --part A
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crisismmd import experiment, plotting  # noqa: E402

plt = plotting.plt
NAMES = {
    "majority": "Majority class", "text_textcnn": "Text only (TextCNN)", "text_bilstm": "Text only (BiLSTM)",
    "image": "Image only (ResNet18)", "fusion_textcnn": "Fusion (TextCNN + image)",
    "fusion_bilstm": "Fusion (BiLSTM + image)",
}
# Fixed colour per model (reference palette), so a model keeps its colour in every figure and part.
# Shared models use the same colours as the group's Part A notebook figures.
MODEL_COLORS = {
    "majority": "#2a78d6", "text_textcnn": "#eb6834", "image": "#1baf7a", "fusion_textcnn": "#eda100",
    "fusion_bilstm": "#e87ba4", "text_bilstm": "#6250d6", "fusion_textcnn_weighted_loss": "#008300",
    "fusion_textcnn_weighted_sampler": "#e34948",
}
STAGES = ["development", "final_training", "evaluation", "part_c_extension", "artefact_production"]
DONE = {"completed", "early-stopped"}


def display_name(key):
    for suffix, text in (("_weighted_loss", " + weighted loss"), ("_weighted_sampler", " + weighted sampler")):
        if key.endswith(suffix):
            return NAMES.get(key[: -len(suffix)], key[: -len(suffix)]) + text
    return NAMES.get(key, key)


def model_order(key):
    base = ["majority", "text_textcnn", "text_bilstm", "image", "fusion_textcnn", "fusion_bilstm"]
    stem = key.replace("_weighted_loss", "").replace("_weighted_sampler", "")
    return (base.index(stem) if stem in base else len(base), key != stem, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", required=True, help="coursework part, e.g. A or B")
    parser.add_argument("--run-log", default="logs/run_log.csv")
    args = parser.parse_args()
    experiment.enter_repo_root()
    out = Path(f"results/environment/part{args.part}")
    out.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()

    log = pd.read_csv(args.run_log, dtype={"experiment_id": str, "evaluated_experiment_id": str})
    part = log[log["coursework_part"].astype(str) == args.part].set_index("experiment_id")
    if part.empty:
        raise SystemExit(f"No records for part {args.part} in {args.run_log}")

    # ---- Figure 1: energy by workflow stage ------------------------------------------------------
    stages = (part.groupby("workflow_stage")
              .agg(runs=("run_status", "size"), runtime=("runtime_seconds", "sum"), energy=("energy_kwh", "sum"),
                   low=("energy_kwh_low", "sum"), high=("energy_kwh_high", "sum"),
                   emissions=("emissions_kgco2e", "sum"))
              .reindex([s for s in STAGES if s in set(part["workflow_stage"])]).sort_values("energy"))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(stages) + 1.4))
    wh, y = stages["energy"] * 1000, np.arange(len(stages))
    ax.barh(y, wh, height=0.55, color=plotting.TEXT_2)
    ax.errorbar(wh, y, xerr=[wh - stages["low"] * 1000, stages["high"] * 1000 - wh], fmt="none",
                ecolor=plotting.MUTED, elinewidth=1, capsize=3)
    for i, (_, r) in enumerate(stages.iterrows()):
        ax.text(r["high"] * 1000, i, f"  {r['energy'] * 1000:.2f} Wh", va="center", fontsize=8, color=plotting.TEXT_2)
        ax.text(r["high"] * 1000, i - 0.3, f"  {int(r['runs'])} run{'s' if r['runs'] != 1 else ''}, {r['runtime']:.0f} s", va="center",
                fontsize=7, color=plotting.MUTED)
    ax.set_yticks(y, [s.replace("_", " ") for s in stages.index])
    ax.set_xlim(0, stages["high"].max() * 1000 * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Energy (Wh), whiskers = low/high uncertainty range")
    total = stages.sum()
    ax.set_title(f"Part {args.part} energy by workflow stage: total {total['energy'] * 1000:.2f} Wh "
                 f"({total['low'] * 1000:.2f}–{total['high'] * 1000:.2f}), {total['emissions'] * 1000:.2f} gCO₂e")
    fig.tight_layout()
    plotting.save(fig, out / "energy_by_stage.png")

    # ---- Per-model energy: final training run + its linked test evaluation, per seed --------------
    train = part[(part["activity"] == "training") & (part["run_type"] == "final") & part["run_status"].isin(DONE)]
    evals = part[(part["activity"] == "evaluation") & (part["run_status"] == "completed")]
    evals = evals.reset_index().drop_duplicates("evaluated_experiment_id", keep="last").set_index("evaluated_experiment_id")
    rows = []
    for run_id, r in train.iterrows():
        if run_id not in evals.index:
            print(f"warning: {run_id} has no completed test evaluation; skipped")
            continue
        e = evals.loc[run_id]
        row = {"model": r["model"], "seed": r["seed"], "test_f1": e["test_macro_f1"],
               "measured_s": np.nansum([r["tracker_duration_seconds"], e["tracker_duration_seconds"]]),
               "cpu_model": r["cpu_model"]}
        for col in ("energy_kwh", "energy_kwh_low", "energy_kwh_high", "emissions_kgco2e",
                    "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"):
            row[col] = np.nansum([r[col], e[col]])
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise SystemExit("No final training runs with linked test evaluations; run scripts/evaluate_final.py first")
    per_model = per_seed.groupby("model").agg(
        seeds=("seed", "size"), f1=("test_f1", "mean"), f1_std=("test_f1", "std"), energy=("energy_kwh", "mean"),
        low=("energy_kwh_low", "mean"), high=("energy_kwh_high", "mean"), emissions=("emissions_kgco2e", "mean"),
        cpu=("cpu_energy_kwh", "mean"), gpu=("gpu_energy_kwh", "mean"), ram=("ram_energy_kwh", "mean"),
        measured_s=("measured_s", "mean"), total_energy=("energy_kwh", "sum"))
    per_model = per_model.loc[sorted(per_model.index, key=model_order)]
    keys = list(per_model.index)
    spare = iter(c for c in plotting.SERIES if c not in MODEL_COLORS.values())
    colors = {k: MODEL_COLORS.get(k) or next(spare, plotting.MUTED) for k in keys}
    names = [display_name(k) for k in keys]
    y = np.arange(len(keys))

    # ---- Figure 2: energy per trained model, and performance against energy ----------------------
    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(13, 0.45 * len(keys) + 2.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax_e.barh(y, per_model["energy"] * 1000, height=0.6, color=[colors[k] for k in keys])
    ax_e.errorbar(per_model["energy"] * 1000, y, xerr=[(per_model["energy"] - per_model["low"]) * 1000,
                                                       (per_model["high"] - per_model["energy"]) * 1000],
                  fmt="none", ecolor=plotting.TEXT_2, elinewidth=1, capsize=3)
    ax_e.set_yticks(y, names)
    ax_e.invert_yaxis()
    ax_e.grid(axis="y", visible=False)
    ax_e.set_xlabel("Energy per seed, training + test evaluation (Wh)")
    ax_e.set_title("Energy per trained model")
    for k in keys:
        r = per_model.loc[k]
        ax_s.errorbar(r["energy"] * 1000, r["f1"], yerr=0 if pd.isna(r["f1_std"]) else r["f1_std"],
                      xerr=[[(r["energy"] - r["low"]) * 1000], [(r["high"] - r["energy"]) * 1000]], fmt="o",
                      color=colors[k], markersize=8, markeredgecolor=plotting.SURFACE, markeredgewidth=1.5,
                      elinewidth=1, capsize=2, label=display_name(k))
    ax_s.set_xlabel("Energy per seed (Wh)")
    ax_s.set_ylabel("Test macro-F1 (mean ± std)")
    ax_s.set_title("Performance against energy")
    ax_s.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plotting.save(fig, out / "energy_vs_performance.png")

    # ---- Figure 3: energy by component, with the idle-machine equivalent ----------------------------
    cpu_name = per_seed["cpu_model"].iloc[0]
    idle_runs = part[(part["activity"] == "idle_baseline") & part["run_status"].isin(DONE)]
    if idle_runs.empty:  # an idle baseline may be filed under another part
        idle_runs = log[(log["activity"] == "idle_baseline") & log["run_status"].isin(DONE)].set_index("experiment_id")
    idle_runs = idle_runs[idle_runs["cpu_model"] == cpu_name].sort_values("started_at")
    idle = idle_runs.iloc[-1] if not idle_runs.empty else None
    if idle is None:
        print(f"warning: no idle-baseline record for {cpu_name}; idle markers omitted")
    idle_w = {c: (idle[f"{c}_power_w_avg"] or 0) if idle is not None else None for c in ("cpu", "gpu", "ram")}
    idle_total_w = idle["energy_kwh"] * 3.6e6 / idle["tracker_duration_seconds"] if idle is not None else None
    has_gpu = per_model["gpu"].fillna(0).sum() > 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(keys) + 1.8), sharey=True)
    titles = {"cpu": "CPU (measured, RAPL)", "gpu": "GPU (measured, NVML)" if has_gpu else "GPU (none on this machine)",
              "ram": "RAM (estimated, CodeCarbon model)"}
    for ax, comp in zip(axes, ("cpu", "gpu", "ram")):
        ax.set_title(titles[comp])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Wh per seed")
        if comp == "gpu" and not has_gpu:
            ax.text(0.5, 0.5, "CPU-only runs:\nno GPU energy", transform=ax.transAxes, ha="center", va="center",
                    color=plotting.MUTED, fontsize=9)
            ax.set_xticks([])
            continue
        ax.barh(y, per_model[comp] * 1000, height=0.6, color=[colors[k] for k in keys])
        if idle is not None:
            idle_wh = idle_w[comp] * per_model["measured_s"] / 3600  # W x s -> Wh
            ax.plot(idle_wh, y, "|", color=plotting.TEXT, markersize=16, markeredgewidth=2)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle("Energy per seed by component.  ┃ = the idle machine's draw over the same measured time",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, out / "energy_by_component.png")

    # ---- Indicators table ---------------------------------------------------------------------------
    base = per_model.loc["majority"] if "majority" in per_model.index else None
    table = pd.DataFrame({
        "Model": names, "Seeds": per_model["seeds"].astype(int).to_numpy(),
        "Test macro-F1": per_model["f1"].round(4).to_numpy(),
        "Energy per seed (kWh)": per_model["energy"].to_numpy(),
        "Energy, all seeds (kWh)": per_model["total_energy"].to_numpy(),
        "Emissions per seed (kgCO2e)": per_model["emissions"].to_numpy(),
        "Macro-F1 per kWh": [None if k == "majority" else r["f1"] / r["energy"] for k, r in per_model.iterrows()],
        "Macro-F1 per kgCO2e": [None if k == "majority" else r["f1"] / r["emissions"] for k, r in per_model.iterrows()],
        "Marginal gCO2e per +0.01 macro-F1 vs majority": [
            None if base is None or k == "majority" else
            1000 * (r["emissions"] - base["emissions"]) / ((r["f1"] - base["f1"]) * 100)
            for k, r in per_model.iterrows()],
        "Idle-draw share of energy": [
            None if idle_total_w is None or k == "majority" else idle_total_w * r["measured_s"] / 3600 / (r["energy"] * 1000)
            for k, r in per_model.iterrows()],
    })
    table.to_csv(out / "environmental_indicators.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))
    if idle is not None:
        print(f"Idle baseline {idle.name}: {idle_total_w:.1f} W (CPU {idle_w['cpu']:.1f} W, "
              f"GPU {idle_w['gpu']:.1f} W, RAM {idle_w['ram']:.1f} W)")
    print(f"Figures -> {out}/")


if __name__ == "__main__":
    main()
