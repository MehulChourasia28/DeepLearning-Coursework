#!/usr/bin/env python3
"""Summarise completed runs for one task: mean ± std over seeds, median-run selection, figures.

For each model configuration (model + text encoder + imbalance method):
  * mean ± sample std (ddof=1) over seeds of validation and test metrics, parameters, time and energy;
  * the run with the MEDIAN validation macro-F1 is selected (before looking at test results) and its
    confusion matrix is plotted;
  * learning curves for every seed.
For Part A it also selects examples where fusion helps (fusion right, unimodal models wrong) or fails
(fusion wrong, a unimodal model right), using the selected runs.

Output: results/summary/<task>/<run-type>/
  runs.csv, summary.csv, summary.md, selected_runs.json, literature_comparison.md,
  model_comparison.png, learning_curves_<config>.png, confusion_<config>.png,
  fusion_examples.csv / fusion_examples.png (Part A)

Example:
  python scripts/summarise.py --task informative --run-type final
"""
import argparse
import html
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crisismmd import data, plotting, tracking  # noqa: E402
from crisismmd.text import fix_mojibake  # noqa: E402

DONE = {"completed", "early-stopped"}
ORDER = ["majority", "text", "image", "fusion"]
# Ofli et al. (2020), test accuracy. Literature values: kept separate from our runs.
LITERATURE = {"informative": {"Text CNN": 0.808, "Image VGG16": 0.833, "Text + image fusion": 0.844},
              "humanitarian": {"Text CNN": 0.704, "Image VGG16": 0.768, "Text + image fusion": 0.784}}


def config_key(run):
    parts = [run["model"]]
    if run.get("text_encoder"):
        parts.append(run["text_encoder"])
    if run.get("imbalance", "none") != "none":
        parts.append(run["imbalance"])
    return "_".join(parts)


def config_label(run):
    name = {"majority": "Majority baseline", "text": "Text", "image": "Image", "fusion": "Fusion"}[run["model"]]
    if run.get("text_encoder"):
        name += " (" + {"textcnn": "TextCNN", "bilstm": "BiLSTM"}[run["text_encoder"]] + ")"
    if run.get("imbalance", "none") != "none":
        name += " + " + run["imbalance"].replace("_", " ")
    return name


def load_runs(results_dir, task, run_type, member):
    runs = []
    for path in sorted((results_dir / "runs").glob("*/run.json")):
        run = json.loads(path.read_text())
        if (run.get("task") != task or run.get("run_type") != run_type or run.get("status") not in DONE
                or (member and run.get("member") != member)):
            continue
        run["dir"] = path.parent
        run["metrics"] = json.loads((path.parent / "metrics.json").read_text())
        run["config_key"], run["config_label"] = config_key(run), config_label(run)
        runs.append(run)
    # If a configuration/seed was run more than once, keep the latest and report the others.
    latest = {}
    for run in runs:
        key = (run["config_key"], run["seed"])
        if key in latest:
            print(f"note: {latest[key]['experiment_id']} superseded by {run['experiment_id']} "
                  f"(same config and seed); only the latest is summarised")
        latest[key] = run
    encoders = [None, "textcnn", "bilstm"]
    imbalances = ["none", "weighted_loss", "weighted_sampler"]
    return sorted(latest.values(), key=lambda r: (ORDER.index(r["model"]), encoders.index(r.get("text_encoder")),
                                                  imbalances.index(r.get("imbalance", "none")), r["seed"]))


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def fmt(mean, std, digits=3):
    return "—" if mean is None else f"{mean:.{digits}f} ± {std:.{digits}f}"


def median_run(runs):
    """Run with the median validation macro-F1 (lower middle for an even count)."""
    ranked = sorted(runs, key=lambda r: r["metrics"]["valid"]["macro_f1"])
    return ranked[(len(ranked) - 1) // 2]


def summarise_config(runs, task, split):
    first = runs[0]
    row = {"config": first["config_key"], "model": first["config_label"], "n_seeds": len(runs),
           "seeds": " ".join(str(r["seed"]) for r in runs),
           "experiment_ids": " ".join(r["experiment_id"] for r in runs),
           "trainable_params": first.get("trainable_params")}
    metric_names = ["macro_f1", "accuracy", "weighted_f1"]
    if task in data.POSITIVE_LABEL:
        positive = data.POSITIVE_LABEL[task]
        metric_names += [f"{positive}_precision", f"{positive}_recall"]
    for name in metric_names:
        for s in ("valid", "test"):
            row[f"{s}_{name}_mean"], row[f"{s}_{name}_std"] = mean_std(
                [r["metrics"].get(s, {}).get(name) for r in runs])
    for label in data.TASK_LABELS[task]:
        row[f"{split}_f1_{label}_mean"], row[f"{split}_f1_{label}_std"] = mean_std(
            [r["metrics"][split]["per_class"][label]["f1"] for r in runs])
        row[f"{split}_correct_{label}"] = " ".join(
            f"{r['metrics'][split]['per_class'][label]['correct']}/{r['metrics'][split]['per_class'][label]['support']}"
            for r in runs)
    for name in ("epochs_completed", "best_epoch", "train_time_s", "energy_kwh", "emissions_kgco2e"):
        row[f"{name}_mean"], row[f"{name}_std"] = mean_std([r.get(name) for r in runs])
    row["energy_kwh_total"] = sum(r.get("energy_kwh") or 0 for r in runs)
    row["selected_run"] = median_run(runs)["experiment_id"]
    return row


def write_markdown(rows, task, split, path):
    positive = data.POSITIVE_LABEL.get(task)
    header = ["Model", "Seeds", "Val macro-F1", f"{split.title()} macro-F1", f"{split.title()} accuracy",
              f"{split.title()} weighted-F1"]
    if positive:
        header += [f"{split.title()} {positive} P", f"{split.title()} {positive} R"]
    header += ["Params", "Epochs (best)", "Train time (s)", "Energy/run (kWh)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model"], str(row["n_seeds"]), fmt(row["valid_macro_f1_mean"], row["valid_macro_f1_std"]),
                 fmt(row[f"{split}_macro_f1_mean"], row[f"{split}_macro_f1_std"]),
                 fmt(row[f"{split}_accuracy_mean"], row[f"{split}_accuracy_std"]),
                 fmt(row[f"{split}_weighted_f1_mean"], row[f"{split}_weighted_f1_std"])]
        if positive:
            cells += [fmt(row[f"{split}_{positive}_precision_mean"], row[f"{split}_{positive}_precision_std"]),
                      fmt(row[f"{split}_{positive}_recall_mean"], row[f"{split}_{positive}_recall_std"])]
        cells += [f"{row['trainable_params']:,}" if row["trainable_params"] else "0",
                  "—" if row["epochs_completed_mean"] in (None, 0) else
                  f"{row['epochs_completed_mean']:.1f} ({row['best_epoch_mean']:.1f})",
                  fmt(row["train_time_s_mean"], row["train_time_s_std"], 1),
                  "—" if row["energy_kwh_mean"] is None else f"{row['energy_kwh_mean']:.2e}"]
        lines.append("| " + " | ".join(cells) + " |")
    per_class = ["", f"Per-class {split} F1 (mean ± std) and number correct per seed:", "",
                 "| Model | " + " | ".join(data.TASK_LABELS[task]) + " |",
                 "|" + "---|" * (len(data.TASK_LABELS[task]) + 1)]
    for row in rows:
        per_class.append("| " + row["model"] + " | " + " | ".join(
            f"{fmt(row[f'{split}_f1_{label}_mean'], row[f'{split}_f1_{label}_std'])} ({row[f'{split}_correct_{label}']})"
            for label in data.TASK_LABELS[task]) + " |")
    notes = ["", f"Mean ± sample standard deviation (ddof=1) over seeds. Confusion matrices use the run with the "
             f"median validation macro-F1 (`selected_run` in summary.csv)."]
    path.write_text("\n".join(lines + per_class + notes) + "\n")


def write_literature(rows, task, split, path):
    lines = [f"# Comparison with Ofli et al. (2020) — accuracy only", "",
             "Published values use pretrained word vectors (text CNN) and VGG16 (image); they are literature",
             "values, not our runs. Our macro-F1 is reported separately in summary.md.", "",
             "| Source | Model | Accuracy |", "|---|---|---|"]
    lines += [f"| Ofli et al. (2020) | {name} | {value:.1%} |" for name, value in LITERATURE[task].items()]
    lines += [f"| Ours ({split}, mean ± std) | {row['model']} | "
              + ("—" if row[f"{split}_accuracy_mean"] is None else
                 f"{row[f'{split}_accuracy_mean']:.1%} ± {row[f'{split}_accuracy_std']:.1%}") + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n")


def read_predictions(run, split):
    return pd.read_csv(run["dir"] / f"predictions_{split}.csv", dtype={"tweet_id": str, "image_id": str})


def lenient_mojibake_fix(text):
    """Display-only repair: also handles bytes undefined in cp1252 (e.g. the end of 'â€\\x9d') and drops
    orphaned bytes whose partner was lost upstream (e.g. 'Â ' where a non-breaking space became a space).

    Not used for training, so the text the models saw stays exactly as in crisismmd.text.
    """
    if not re.search("[âÃÂð]", text):
        return text
    try:
        raw = b"".join(ch.encode("cp1252") if ch not in "\x81\x8d\x8f\x90\x9d" else ch.encode("latin-1")
                       for ch in text)
    except UnicodeError:  # contains characters outside cp1252, so it was not simply mis-decoded
        return fix_mojibake(text)
    return raw.decode("utf-8", errors="ignore")


def display_text(text, width=48, max_lines=5):
    text = html.unescape(lenient_mojibake_fix(text))
    text = re.sub(r"https?://\S+", "", text)
    text = "".join(ch for ch in text if ord(ch) < 0x2000 or ch in "’‘“”—–…")  # drop glyphs fonts lack (emoji)
    lines = textwrap.wrap(text.strip(), width)
    return "\n".join(lines[:max_lines]) + (" …" if len(lines) > max_lines else "")


def fusion_examples(selected, task, split, out_dir, fusion_key, data_root, per_side=3):
    """Pick examples where fusion helps or fails relative to the matching text-only and image-only runs."""
    encoder = fusion_key.split("_")[1]
    needed = {"fusion": fusion_key, "text": f"text_{encoder}", "image": "image"}
    if any(key not in selected for key in needed.values()):
        print(f"fusion examples skipped: need configs {list(needed.values())}")
        return
    merged = None
    for name, key in needed.items():
        frame = read_predictions(selected[key], split)
        prob = frame[[c for c in frame.columns if c.startswith("prob_")]]
        frame = frame[["tweet_id", "image_id", "event_name", "image", "tweet_text", "label"]].assign(
            **{f"{name}_pred": frame["pred"], f"{name}_correct": frame["correct"],
               f"{name}_p_true": [prob.iloc[i][f"prob_{label}"] for i, label in enumerate(frame["label"])]})
        merged = frame if merged is None else merged.merge(
            frame[["tweet_id", "image_id", f"{name}_pred", f"{name}_correct", f"{name}_p_true"]],
            on=["tweet_id", "image_id"], validate="one_to_one")
    f, t, i = merged["fusion_correct"], merged["text_correct"], merged["image_correct"]
    merged["case"] = np.select(
        [f & ~t & ~i, f & (~t | ~i), ~f & t & i, ~f & (t | i)],
        ["helps: both unimodal wrong", "helps: one unimodal wrong", "fails: both unimodal right",
         "fails: one unimodal right"], default="")
    counts = merged["case"].replace("", "all agree or all wrong").value_counts().to_dict()
    picks = []
    for side, cases in (("helps", ["helps: both unimodal wrong", "helps: one unimodal wrong"]),
                        ("fails", ["fails: both unimodal right", "fails: one unimodal right"])):
        pool = merged[merged["case"].isin(cases)].copy()
        pool["priority"] = pool["case"].map({c: n for n, c in enumerate(cases)})
        # Most confident cases first: fusion's probability of the true class (helps) or of its wrong answer.
        pool["confidence"] = pool["fusion_p_true"] if side == "helps" else 1 - pool["fusion_p_true"]
        pool = pool.sort_values(["priority", "confidence"], ascending=[True, False])
        pool = pool.drop_duplicates("tweet_id")  # avoid the same tweet twice with different images
        picks.append(pool.head(per_side))
    chosen = pd.concat(picks)
    chosen.drop(columns=["priority"]).to_csv(out_dir / "fusion_examples.csv", index=False)
    merged["case"].replace("", "all agree or all wrong").value_counts().rename_axis("case").to_csv(
        out_dir / "fusion_case_counts.csv")
    print(f"fusion example cases ({split}): {counts}")

    from PIL import Image
    plotting.apply_style()
    if chosen.empty:
        print("no helps/fails examples found")
        return
    columns = -(-len(chosen) // 2)  # ceil
    fig = plotting.plt.figure(figsize=(4.4 * columns, 12.5))
    grid = fig.add_gridspec(4, columns, height_ratios=[3, 1.55, 3, 1.55], hspace=0.12, wspace=0.08)
    for index, (_, row) in enumerate(chosen.iterrows()):
        r, c = divmod(index, columns)
        image_ax, text_ax = fig.add_subplot(grid[2 * r, c]), fig.add_subplot(grid[2 * r + 1, c])
        with Image.open(data_root / row["image"]) as image:
            image.thumbnail((448, 448))
            image_ax.imshow(image.convert("RGB"))
        helps = row["case"].startswith("helps")
        image_ax.set_title(row["case"].replace(":", " —"), fontsize=10, loc="left",
                           color=plotting.SERIES[0] if helps else plotting.SERIES[1])
        verdict = lambda name: f"{'✓' if row[f'{name}_correct'] else '✗'} {row[f'{name}_pred'].replace('_', ' ')}"
        text_ax.text(0, 1, f"{display_text(row['tweet_text'])}", va="top", fontsize=8.5, color=plotting.TEXT,
                     transform=text_ax.transAxes, style="italic")
        text_ax.text(0, 0.38, f"true: {row['label'].replace('_', ' ')}  ·  {row['event_name'].replace('_', ' ')}\n"
                              f"text {verdict('text')}   image {verdict('image')}\n"
                              f"fusion {verdict('fusion')}  (p(true) = {row['fusion_p_true']:.2f})",
                     va="top", fontsize=8.5, color=plotting.TEXT_2, transform=text_ax.transAxes)
        for ax in (image_ax, text_ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        text_ax.set_facecolor(plotting.SURFACE)
    fig.suptitle(f"Where fusion helps and fails ({split} split, selected runs)", x=0.12, y=0.93, ha="left",
                 fontsize=13, fontweight="semibold")
    plotting.save(fig, out_dir / "fusion_examples.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(data.TASK_LABELS))
    parser.add_argument("--run-type", default="final", choices=["debug", "exploratory", "final"])
    parser.add_argument("--member", default=None, help="only include this member's runs")
    parser.add_argument("--results-dir", type=Path, default=tracking.RESULTS_DIR)
    parser.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    parser.add_argument("--fusion-config", default="fusion_textcnn",
                        help="fusion config used for the helps/fails examples (Part A)")
    args = parser.parse_args()

    runs = load_runs(args.results_dir, args.task, args.run_type, args.member)
    if not runs:
        raise SystemExit(f"no completed {args.run_type} runs for task {args.task}")
    split = "test" if all("test" in r["metrics"] for r in runs) else "valid"
    out_dir = args.results_dir / "summary" / args.task / args.run_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs; reporting on the {split} split -> {tracking.portable(out_dir)}")

    groups = {}
    for run in runs:
        groups.setdefault(run["config_key"], []).append(run)
    rows = [summarise_config(group, args.task, split) for group in groups.values()]
    selected = {key: median_run(group) for key, group in groups.items()}

    pd.DataFrame([{"experiment_id": r["experiment_id"], "config": r["config_key"], "seed": r["seed"],
                   "best_epoch": r.get("best_epoch"), "epochs_completed": r.get("epochs_completed"),
                   "val_macro_f1": r["metrics"]["valid"]["macro_f1"],
                   "test_macro_f1": r["metrics"].get("test", {}).get("macro_f1"),
                   "test_accuracy": r["metrics"].get("test", {}).get("accuracy"),
                   "trainable_params": r.get("trainable_params"), "train_time_s": r.get("train_time_s"),
                   "energy_kwh": r.get("energy_kwh"), "selected": selected[r["config_key"]] is r}
                  for r in runs]).to_csv(out_dir / "runs.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    write_markdown(rows, args.task, split, out_dir / "summary.md")
    write_literature(rows, args.task, split, out_dir / "literature_comparison.md")
    tracking.write_json(out_dir / "selected_runs.json",
                        {key: {"experiment_id": run["experiment_id"], "seed": run["seed"],
                               "val_macro_f1": run["metrics"]["valid"]["macro_f1"],
                               "rule": "median validation macro-F1 over seeds"} for key, run in selected.items()})

    majority = next((g for k, g in groups.items() if k == "majority"), None)
    learned = [(g[0]["config_label"], [r["metrics"][split]["macro_f1"] for r in g])
               for k, g in groups.items() if k != "majority"]
    if learned:
        reference = ("majority baseline", majority[0]["metrics"][split]["macro_f1"]) if majority else None
        plotting.model_comparison(learned, f"{split.title()} macro-F1", out_dir / "model_comparison.png", reference)
    for key, group in groups.items():
        histories = {r["seed"]: pd.read_csv(r["dir"] / "history.csv") for r in group
                     if (r["dir"] / "history.csv").exists()}
        if histories:
            plotting.learning_curves(histories, f"{group[0]['config_label']} — {args.task}",
                                     out_dir / f"learning_curves_{key}.png")
        run = selected[key]
        plotting.confusion_matrix(run["metrics"][split]["confusion_matrix"], data.TASK_LABELS[args.task],
                                  f"{group[0]['config_label']} — {split} confusion matrix\n"
                                  f"{run['experiment_id']} (seed {run['seed']}, median val macro-F1)",
                                  out_dir / f"confusion_{key}.png")
    if args.task == "informative":
        fusion_examples(selected, args.task, split, out_dir, args.fusion_config, args.data_root)
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
