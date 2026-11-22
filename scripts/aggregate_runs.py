"""Collect every logs/runs/*/summary.json into the submitted experiment log.

Usage (from the coursework directory):
    python -m scripts.aggregate_runs [--root logs/runs] [--out logs]

Writes:
    run_log.csv             One row per run: the compulsory experiment log
    stage_summary.csv       Table 1: runs, runtime, energy, emissions by workflow stage
    model_summary.csv       Table 2: final models across seeds, F1 and environmental cost
    aggregate_summary.json  Run counts, totals with ranges, wasted-compute ratio, warnings

Each run is counted once, in its own workflow stage. Evaluation energy is
linked to the training run through evaluated_experiment_id for Table 2 only.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

LEADING_COLUMNS = [
    "experiment_id", "coursework_part", "task", "run_type", "original_run_type",
    "activity", "workflow_stage", "model", "seed",
    "training_examples", "dataset_fraction", "training_examples_processed", "examples_processed",
    "epochs_completed", "selected_epoch", "stopping_point", "batch_size",
    "execution_device", "cpu_model", "gpu_model", "device_count", "ram_gb",
    "runtime_seconds", "tracker_duration_seconds",
    "energy_kwh", "energy_kwh_low", "energy_kwh_high",
    "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
    "cpu_energy_status", "gpu_energy_status", "ram_energy_status",
    "carbon_intensity_gco2e_per_kwh", "carbon_intensity_type", "carbon_intensity_source",
    "emissions_kgco2e", "emissions_kgco2e_low", "emissions_kgco2e_high",
    "f1_scale", "validation_macro_f1", "test_macro_f1",
    "run_status", "failure_reason",
    "cache_used", "cache_path", "checkpoint_used", "checkpoint_input_path",
    "evaluated_experiment_id", "reuse_notes", "rerun_of", "rerun_reason",
    "trainable_parameters", "measurement_method_id", "measurement_method",
    "measurement_boundary", "missing_measurements", "started_at", "finished_at",
]
STAGES = ["development", "final_training", "evaluation", "part_c_extension", "artefact_production"]
ENERGY_KEYS = ["energy_kwh", "energy_kwh_low", "energy_kwh_high",
               "emissions_kgco2e", "emissions_kgco2e_low", "emissions_kgco2e_high"]


def load_runs(root):
    runs = []
    for path in sorted(Path(root).glob("*/summary.json")):
        runs.append(json.loads(path.read_text()))
    return sorted(runs, key=lambda r: r.get("started_at") or "")


def _cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return "" if value is None else value


def _sum(runs, key):
    values = [r.get(key) for r in runs]
    return sum(v for v in values if v is not None)


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})


def aggregate(root="logs/runs", out="logs"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    warnings = []
    for r in runs:
        if r.get("run_status") in {"registered", "running"}:
            warnings.append(f"{r['experiment_id']}: status '{r['run_status']}' (process crashed or still "
                            "running); energy may be missing - keep it and explain in missing_measurements")
        elif r.get("energy_kwh") is None:
            warnings.append(f"{r['experiment_id']}: no energy measurement ({r.get('missing_measurements')})")

    # One row per run.
    extra = sorted({k for r in runs for k in r} - set(LEADING_COLUMNS))
    write_csv(out / "run_log.csv", runs, LEADING_COLUMNS + extra)

    # Table 1: by workflow stage.
    total_energy = _sum(runs, "energy_kwh")
    stage_rows = []
    for stage in STAGES + ["TOTAL"]:
        group = runs if stage == "TOTAL" else [r for r in runs if r.get("workflow_stage") == stage]
        row = {"stage": stage, "runs": len(group), "runtime_seconds": _sum(group, "runtime_seconds"),
               **{k: _sum(group, k) for k in ENERGY_KEYS}}
        row["energy_share"] = row["energy_kwh"] / total_energy if total_energy else None
        stage_rows.append(row)
    write_csv(out / "stage_summary.csv", stage_rows,
              ["stage", "runs", "runtime_seconds", *ENERGY_KEYS, "energy_share"])

    # Table 2: final, completed/early-stopped training runs per model, with linked test evaluations.
    evaluations = defaultdict(list)
    for r in runs:
        if r.get("activity") == "evaluation" and r.get("evaluated_experiment_id"):
            evaluations[r["evaluated_experiment_id"]].append(r)
    models = defaultdict(list)
    for r in runs:
        if (r.get("activity") == "training" and r.get("run_type") == "final"
                and r.get("run_status") in {"completed", "early-stopped"}):
            models[(r.get("coursework_part"), r.get("task"), r.get("model"))].append(r)
    model_rows = []
    for (part, task, model), group in sorted(models.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tests, energy, emissions = [], [], []
        for r in group:
            linked = [e for e in evaluations.get(r["experiment_id"], [])
                      if e.get("run_status") == "completed" and e.get("test_macro_f1") is not None]
            tests.append(linked[-1]["test_macro_f1"] if linked else None)
            energy.append((r.get("energy_kwh") or 0) + sum(e.get("energy_kwh") or 0 for e in linked))
            emissions.append((r.get("emissions_kgco2e") or 0) + sum(e.get("emissions_kgco2e") or 0 for e in linked))
        val_mean, val_std = _mean_std([r.get("validation_macro_f1") for r in group])
        test_mean, test_std = _mean_std(tests)
        energy_mean, _ = _mean_std(energy)
        emissions_mean, _ = _mean_std(emissions)
        model_rows.append({
            "coursework_part": part, "task": task, "model": model,
            "seeds": sorted(r.get("seed") for r in group if r.get("seed") is not None),
            "runs": len(group), "seeds_with_test_eval": sum(t is not None for t in tests),
            "validation_macro_f1_mean": val_mean, "validation_macro_f1_std": val_std,
            "test_macro_f1_mean": test_mean, "test_macro_f1_std": test_std,
            "trainable_parameters": group[0].get("trainable_parameters"),
            "runtime_seconds_mean": _mean_std([r.get("runtime_seconds") for r in group])[0],
            "energy_kwh_total": sum(energy), "energy_kwh_per_seed_mean": energy_mean,
            "emissions_kgco2e_total": sum(emissions), "emissions_kgco2e_per_seed_mean": emissions_mean,
            "test_f1_per_kwh": (test_mean / energy_mean) if test_mean is not None and energy_mean else None,
            "test_f1_per_kgco2e": (test_mean / emissions_mean) if test_mean is not None and emissions_mean else None,
        })
    write_csv(out / "model_summary.csv", model_rows, list(model_rows[0]) if model_rows else
              ["coursework_part", "task", "model"])

    failed_energy = _sum([r for r in runs if r.get("run_type") == "failed"], "energy_kwh")
    summary = {
        "runs": len(runs),
        "runs_by_run_type": {t: sum(r.get("run_type") == t for r in runs)
                             for t in ["debug", "exploratory", "final", "failed"]},
        "runs_by_status": {s: sum(r.get("run_status") == s for r in runs)
                           for s in sorted({r.get("run_status") for r in runs})},
        "epochs_completed_total": _sum(runs, "epochs_completed"),
        "runtime_seconds_total": _sum(runs, "runtime_seconds"),
        **{f"{k}_total": _sum(runs, k) for k in ENERGY_KEYS},
        "failed_energy_kwh": failed_energy,
        "wasted_compute_ratio_failed_over_total": failed_energy / total_energy if total_energy else None,
        "measurement_method_ids": sorted({str(r.get("measurement_method_id")) for r in runs}),
        "warnings": warnings,
        "note": ("Each run is counted once in its own workflow_stage. Add any unreported exploratory "
                 "energy to the wasted-compute numerator by hand if applicable."),
    }
    (out / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="logs/runs")
    parser.add_argument("--out", default="logs")
    args = parser.parse_args()
    summary = aggregate(args.root, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "note"}, indent=2))
    print(f"Wrote run_log.csv, stage_summary.csv, model_summary.csv, aggregate_summary.json to {args.out}/")


if __name__ == "__main__":
    main()
