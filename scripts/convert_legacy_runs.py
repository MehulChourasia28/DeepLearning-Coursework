#!/usr/bin/env python3
"""Convert pre-logger records into the group run-log format (logs/runs/<id>/summary.json).

Sources (no recomputation, nothing retrained):
  results/runs/SHC-A-*/run.json                Part A runs logged by crisismmd/tracking.py (2026-11-14/15)
  data/crisismmd/features/resnet18_*.json      the two ResNet18 feature-extraction runs (2026-11-14)

What changes in conversion:
  * Emissions are recomputed with the group's central carbon intensity (templates/measurement_method.json,
    122 gCO2/kWh); the legacy records used codecarbon's GBR default (237.6). Energy is unchanged.
  * Low/high energy and emission ranges use the same factors as scripts/run_logger.py.
  * Final runs are split into a training record and a linked evaluation record, because the legacy tracker
    measured the two phases separately. For debug/exploratory runs the two phases are summed into one
    training record (validation only).
Each converted record keeps `converted_from` and explains gaps in `missing_measurements`.

Usage:
  python scripts/convert_legacy_runs.py            # writes records that don't exist yet
  python scripts/convert_legacy_runs.py --overwrite
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisismmd import experiment  # noqa: E402
from crisismmd.tracking import write_json  # noqa: E402

LOGGER_FIELDS = [
    "run_status", "original_run_type", "failure_reason", "started_at", "finished_at", "epochs_completed",
    "selected_epoch", "stopping_point", "runtime_seconds", "tracker_duration_seconds", "energy_kwh",
    "energy_kwh_low", "energy_kwh_high", "emissions_kgco2e", "emissions_kgco2e_low", "emissions_kgco2e_high",
    "uncertainty_factors_applied", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "cpu_energy_status",
    "gpu_energy_status", "ram_energy_status", "measurement_method", "cpu_power_w_avg", "gpu_power_w_avg",
    "ram_power_w_avg", "cpu_utilisation_fraction", "gpu_utilisation_fraction", "f1_scale",
    "validation_macro_f1", "test_macro_f1", "checkpoint_path", "predictions_path",
    "training_examples_processed", "examples_processed", "execution_device", "cpu_model", "gpu_model",
    "device_count", "ram_gb", "os", "tracker_name", "tracker_version", "torch_version", "tensorboard_version",
    "trainable_parameters", "cache_output_paths", "missing_measurements",
]
LEGACY_NOTE = "Converted from a pre-logger record ({source}); not re-measured."
LEGACY_METHOD = ("CodeCarbon 3.3.2 OfflineEmissionsTracker, tracking_mode=machine (default); CPU: {cpu}; "
                 "GPU: none (no GPU tracked); RAM: estimated (CodeCarbon heuristic, 5 W per estimated DIMM)")


def base_record(template):
    method_file = "templates/measurement_method.json"
    config = json.loads(Path(template).read_text())
    method = json.loads(Path(method_file).read_text())
    record = {**config, **method, "measurement_method_file": method_file}
    record.update({field: None for field in LOGGER_FIELDS})
    record.update(f1_scale="0-1", training_examples_processed=0, examples_processed=0, cache_output_paths=[],
                  missing_measurements="", tracker_name="CodeCarbon", tracker_version="3.3.2")
    return config, record


def add_uncertainty(record, components, statuses):
    """Same factors and arithmetic as RunLogger._record_uncertainty; components=None means total only."""
    factors = record["component_energy_factors"]
    overhead_lo, overhead_hi = record["unmeasured_overhead_factor"]
    ci_lo, ci_hi = record["carbon_intensity_range_gco2e_per_kwh"]
    applied, low, high = {}, 0.0, 0.0
    if components is None:  # only the total is known: treat it as 'unknown'
        lo, hi = factors["unknown"]
        applied["total"] = {"factor_key": "unknown", "low": lo, "high": hi}
        low, high = record["energy_kwh"] * lo, record["energy_kwh"] * hi
    else:
        for component, energy in components.items():
            status = statuses[component]
            key = "none" if status == "none" else f"{component}_{status}"
            lo, hi = factors.get(key, factors["unknown"])
            applied[component] = {"factor_key": key, "low": lo, "high": hi}
            low += energy * lo
            high += energy * hi
    applied["unmeasured_overhead"] = {"low": overhead_lo, "high": overhead_hi}
    applied["carbon_intensity_gco2e_per_kwh"] = {"low": ci_lo, "high": ci_hi}
    low, high = low * overhead_lo, high * overhead_hi
    record.update(energy_kwh_low=low, energy_kwh_high=high, emissions_kgco2e_low=low * ci_lo / 1000,
                  emissions_kgco2e_high=high * ci_hi / 1000, uncertainty_factors_applied=applied,
                  emissions_kgco2e=record["energy_kwh"] * record["carbon_intensity_gco2e_per_kwh"] / 1000)


def combine(*phases):
    """Sum tracked phases; average power = energy / duration per component."""
    duration = sum(p["duration"] for p in phases)
    out = {"duration": duration}
    for c in ("cpu", "gpu", "ram"):
        energy = sum(p[f"{c}_energy"] for p in phases)
        out[f"{c}_energy"] = energy
        out[f"{c}_power"] = energy * 3_600_000 / duration if duration else None
    out["energy_consumed"] = sum(p["energy_consumed"] for p in phases)
    return out


def fill_energy(record, phase):
    statuses = {"cpu": "measured", "gpu": "none", "ram": "estimated"}
    record.update(
        tracker_duration_seconds=phase["duration"], energy_kwh=phase["energy_consumed"],
        cpu_energy_kwh=phase["cpu_energy"], gpu_energy_kwh=phase["gpu_energy"], ram_energy_kwh=phase["ram_energy"],
        cpu_power_w_avg=phase["cpu_power"], gpu_power_w_avg=phase["gpu_power"], ram_power_w_avg=phase["ram_power"],
        **{f"{c}_energy_status": s for c, s in statuses.items()},
        measurement_method=LEGACY_METHOD.format(cpu="measured (intel_rapl package energy counter)"))
    add_uncertainty(record, {c: phase[f"{c}_energy"] for c in statuses}, statuses)


def convert_training_run(run_dir):
    old = json.loads((run_dir / "run.json").read_text())
    if old.get("status") not in ("completed", "early-stopped"):
        raise SystemExit(f"{old['experiment_id']}: status {old.get('status')} - convert by hand")
    final = old["run_type"] == "final"
    key = experiment.model_key(old["model"], old.get("text_encoder"), old.get("imbalance", "none"))
    majority = old["model"] == "majority"
    source = (run_dir / "run.json").as_posix()
    detail = old["energy_detail"]
    unmeasured_s = max(0.0, old["runtime_s"] - old["train_time_s"] - old["eval_time_s"])
    gaps = [LEGACY_NOTE.format(source=source),
            f"Data loading and tokenisation before the legacy tracker started (~{unmeasured_s:.1f} s) were "
            "neither timed in runtime_seconds nor measured.",
            "CPU/GPU utilisation fractions were not recorded by the legacy tracker.",
            "Emissions recomputed at the central carbon intensity; the legacy record used 237.6 g/kWh."]
    common = dict(
        coursework_part=old["part"], task=experiment.TASK_NAMES[old["task"]], model=key,
        seed=None if majority else old["seed"], training_examples=old["train_examples"],
        dataset_fraction=old["dataset_fraction"], cache_used=old["model"] in ("image", "fusion"),
        cache_path="data/crisismmd/features/resnet18_imagenet1k_v1.npz" if old["model"] in ("image", "fusion") else None,
        execution_device=old["device"], cpu_model=old["cpu_model"], gpu_model="N/A", device_count=1,
        ram_gb=old["ram_gib"], os=old["os"], torch_version=old["torch"],
        trainable_parameters=old["trainable_params"], converted_from=source,
        hyperparameters={"model_key": key, **{k: old["config"].get(k) for k in (
            "max_len", "min_freq", "emb_dim", "hidden", "dropout", "weight_decay")},
            "text_encoder": old.get("text_encoder"), "imbalance": old.get("imbalance"),
            "vocab_size": old.get("vocab_size"), "member": old.get("member"),
            "code": "crisismmd package + scripts/train.py (legacy tracking.py version)"},
    )
    started = datetime.fromisoformat(old["started_at"])

    # Training record.
    _, train = base_record("templates/run_config.json")
    train_phase = detail["train"] if final else combine(detail["train"], detail["eval"])
    train.update(common)
    train.update(
        experiment_id=old["experiment_id"], activity="training",
        workflow_stage=experiment.workflow_stage(old["part"], old["run_type"]), run_type=old["run_type"],
        batch_size=None if majority else old["batch_size"], max_epochs=None if majority else old["max_epochs"],
        early_stopping_patience=None if majority else old["patience"], optimizer=None if majority else "Adam",
        initial_lr=None if majority else old["lr"], scheduler=None, measure_power_secs=15,
        measurement_boundary=(
            "Legacy tracker: model construction, training and per-epoch validation"
            + ("" if final else ", plus best-epoch validation predictions, metrics and prediction CSV (second "
               "tracked phase, summed)")
            + "; excludes data loading/tokenisation (not tracked), feature extraction, dataset download"
            + (" and final evaluation (separate record)." if final else ".") + " " + experiment.MACHINE_NOTE),
        run_status=old["status"], started_at=old["started_at"],
        finished_at=(started + timedelta(seconds=unmeasured_s + old["train_time_s"]
                                         + (0 if final else old["eval_time_s"]))).isoformat(),
        runtime_seconds=old["train_time_s"] + (0 if final else old["eval_time_s"]),
        epochs_completed=old["epochs_completed"], selected_epoch=old.get("best_epoch"),
        stopping_point={"split": "train", "epoch": old["epochs_completed"], "batch": None},
        training_examples_processed=old["epochs_completed"] * old["train_examples"] if not majority else old["train_examples"],
        validation_macro_f1=old["val_macro_f1"],
        checkpoint_path=None if majority else (run_dir / "model.pt").as_posix(),
        predictions_path=(run_dir / "predictions_valid.csv").as_posix(),
        reuse_notes=("Frozen ResNet18 features reused from the cached .npz." if common["cache_used"]
                     else "No image features used by this model."),
        rerun_of=None, rerun_reason=old.get("rerun_reason") or None,
        notes=f"{old.get('purpose') or ''} [legacy ID scheme <member>-<part>-<number>]".strip(),
        missing_measurements="; ".join(gaps + (["checkpoint model.pt is git-ignored (local only)"] if not majority else [])),
    )
    fill_energy(train, train_phase)
    records = [train]

    # Linked evaluation record for final runs.
    if final:
        _, ev = base_record("templates/evaluation_config.json")
        ev.update(common)
        eval_started = datetime.fromisoformat(train["finished_at"])
        ev.update(
            experiment_id=f"{old['experiment_id']}_test_eval", activity="evaluation", workflow_stage="evaluation",
            run_type="final", batch_size=256, measure_power_secs=15,
            measurement_boundary=("Legacy tracker: validation and test prediction with the best-epoch weights "
                                  "held in memory directly after training, metrics and prediction CSVs; "
                                  "excludes training (see evaluated_experiment_id). " + experiment.MACHINE_NOTE),
            run_status="completed", started_at=eval_started.isoformat(),
            finished_at=(eval_started + timedelta(seconds=old["eval_time_s"])).isoformat(),
            runtime_seconds=old["eval_time_s"], examples_processed=1573 + 1534,
            validation_macro_f1=None, test_macro_f1=old["test_macro_f1"],
            evaluated_experiment_id=old["experiment_id"], checkpoint_used=True,
            checkpoint_input_path=train["checkpoint_path"],
            predictions_path=(run_dir / "predictions_test.csv").as_posix(),
            reuse_notes="Best-epoch weights of the linked training run (held in memory; same weights as model.pt).",
            notes="Legacy evaluation phase: validation + test predictions in one tracked phase.",
            missing_measurements="; ".join(gaps[:1] + gaps[2:]),
        )
        fill_energy(ev, detail["eval"])
        records.append(ev)
    return records


def convert_feature_runs(features_dir):
    records = []
    for number, name, run_type in ((1, "resnet18_imagenet1k_v1_limit64.json", "debug"),
                                   (2, "resnet18_imagenet1k_v1.json", "final")):
        path = features_dir / name
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        finished = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        _, rec = base_record("templates/feature_extraction_config.json")
        rec.update(
            experiment_id=f"SHC-F-{number:03d}", run_type=run_type,
            workflow_stage="artefact_production" if run_type == "final" else "development",
            batch_size=meta["batch_size"], measure_power_secs=15,
            measurement_boundary=("Loading pretrained ResNet18 weights, image decoding, prescribed preprocessing, "
                                  "frozen ResNet18 inference and writing the .npz cache for every image referenced "
                                  "by the prepared splits" + (" (first 64 images only)" if run_type == "debug" else "")
                                  + "; excludes dataset download. " + experiment.MACHINE_NOTE),
            run_status="completed", started_at=(finished - timedelta(seconds=meta["runtime_seconds"])).isoformat(),
            finished_at=finished.isoformat(), runtime_seconds=meta["runtime_seconds"],
            tracker_duration_seconds=None, energy_kwh=meta["energy_kwh"], examples_processed=meta["num_images"],
            execution_device=meta["device"], cpu_model="13th Gen Intel(R) Core(TM) i5-13500H", gpu_model="N/A",
            device_count=1, ram_gb=15.2, torch_version=meta["torch"], trainable_parameters=0,
            cpu_energy_status="estimated", gpu_energy_status="none", ram_energy_status="estimated",
            measurement_method=LEGACY_METHOD.format(cpu="estimated (RAPL energy counter unreadable at the time)"),
            cache_output_paths=[f"data/crisismmd/features/{name.replace('.json', '.npz')}"],
            split_examples=None, converted_from=path.as_posix(),
            reuse_notes="Creates the image-feature cache (all 12,708 images; Part B images are a subset).",
            notes="Reconstructed from the feature-cache metadata JSON written by scripts/extract_image_features.py.",
            missing_measurements="; ".join([
                LEGACY_NOTE.format(source=path.as_posix()),
                "Only total energy and runtime were stored: no CPU/GPU/RAM split, average power or utilisation.",
                "CPU energy was estimated by CodeCarbon (RAPL unreadable at the time), so the wide 'unknown' "
                "factor range is applied to the total.",
                "Timestamps reconstructed from the metadata file's modification time.",
                "Emissions recomputed at the central carbon intensity; the legacy record used 237.6 g/kWh."]),
        )
        add_uncertainty(rec, None, None)
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="rewrite records already converted")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written; write nothing")
    args = parser.parse_args()
    experiment.enter_repo_root()
    records = convert_feature_runs(Path("data/crisismmd/features"))
    for run_dir in sorted(Path("results/runs").glob("SHC-A-*")):
        records.extend(convert_training_run(run_dir))
    if args.dry_run:
        for r in records:
            print(f"{r['experiment_id']:24s} {r['activity']:18s} {r['workflow_stage']:19s} {r['run_type']:11s} "
                  f"{str(r['model'])[:28]:28s} E={r['energy_kwh']:.2e} kWh [{r['energy_kwh_low']:.2e}-"
                  f"{r['energy_kwh_high']:.2e}] {r['emissions_kgco2e'] * 1000:.4f} g "
                  f"val={r['validation_macro_f1']} test={r['test_macro_f1']}")
        print(f"{len(records)} records (dry run, nothing written)")
        return
    written = skipped = 0
    for record in records:
        out = Path(experiment.LOG_ROOT) / record["experiment_id"]
        if (out / "summary.json").exists() and not args.overwrite:
            skipped += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", {k: v for k, v in record.items() if k not in LOGGER_FIELDS})
        write_json(out / "summary.json", record)
        written += 1
    print(f"{written} records written, {skipped} already present, in {experiment.LOG_ROOT}/")
    print("Next: python -m scripts.aggregate_runs")


if __name__ == "__main__":
    main()
