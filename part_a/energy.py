"""One environmental/experiment record per run (guidance sections 2, 3 and 5).

Usage:
    with RunLogger(run_type="final", workflow_stage="final_training", ...) as run:
        run.mark_training_start()
        ...train...
        run.update(validation_macro_f1=0.81, epochs_completed=7)
    # run.json written, logs/part_a/run_log.csv rebuilt.

The record is registered (written with status "running") before tracking starts.
Failed and interrupted runs are kept: run_type becomes "failed" and the original
purpose is stored in original_run_type.
"""
import csv
import json
import logging
import os
import platform
import threading
import time
import traceback
from datetime import datetime, timezone

import psutil

from . import config as C

RUN_TYPES = {"debug", "exploratory", "final", "failed"}
STAGES = {"development", "final_training", "evaluation", "part_c_extension", "artefact_production"}

# Column order of the submitted experiment log. Compulsory fields first.
LOG_COLUMNS = [
    "experiment_id", "coursework_part", "task", "run_type", "original_run_type", "activity",
    "workflow_stage", "model", "seed", "training_examples", "dataset_fraction",
    "examples_processed", "epochs_completed", "selected_epoch", "stopping_point", "batch_size",
    "execution_device", "cpu_model", "gpu_model", "device_count", "ram_gb",
    "runtime_seconds", "setup_seconds", "loop_seconds", "energy_kwh", "energy_kwh_low",
    "energy_kwh_high", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "cpu_energy_status",
    "gpu_energy_status", "ram_energy_status", "engineering_estimate_kwh",
    "cpu_utilisation_fraction", "carbon_intensity_gco2e_per_kwh", "carbon_intensity_type",
    "emissions_kgco2e", "emissions_kgco2e_low", "emissions_kgco2e_high", "f1_scale",
    "validation_macro_f1", "test_macro_f1", "run_status", "failure_reason", "cache_used",
    "cache_path", "checkpoint_used", "checkpoint_input_path", "evaluated_experiment_id",
    "reuse_notes", "rerun_of", "rerun_reason", "trainable_parameters", "checkpoint_path",
    "predictions_path", "measurement_method_id", "tracker_name", "tracker_version",
    "measurement_boundary", "missing_measurements", "started_at", "finished_at", "notes",
]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_experiment_id():
    C.RUNS.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.rsplit("-", 1)[1]) for p in C.RUNS.glob(f"{C.RUN_PREFIX}-*")
               if p.name.rsplit("-", 1)[1].isdigit()]
    return f"{C.RUN_PREFIX}-{max(numbers, default=0) + 1:03d}"


def hardware_info():
    return {
        "execution_device": "cpu",
        "cpu_model": _cpu_name(),
        "gpu_model": "N/A",
        "device_count": 1,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
    }


def _cpu_name():
    try:
        from cpuinfo import get_cpu_info  # installed with codecarbon
        return get_cpu_info().get("brand_raw") or platform.processor()
    except Exception:
        return platform.processor()


class _UtilSampler(threading.Thread):
    """Samples whole-machine CPU utilisation once per second (for the 5.2 estimate)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self._halt = [], threading.Event()
        psutil.cpu_percent(None)

    def run(self):
        while not self._halt.wait(1.0):
            self.samples.append(psutil.cpu_percent(None) / 100.0)

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        return sum(self.samples) / len(self.samples) if self.samples else None


class RunLogger:
    def __init__(self, *, run_type, workflow_stage, activity, model, measurement_boundary,
                 seed=None, training_examples=None, dataset_fraction=None, batch_size=None,
                 cache_used=False, cache_path="", checkpoint_used=False,
                 checkpoint_input_path="", evaluated_experiment_id="", reuse_notes="",
                 rerun_of="", rerun_reason="", notes="", hyperparameters=None):
        if run_type not in RUN_TYPES - {"failed"}:
            raise ValueError(f"run_type must be debug/exploratory/final, got {run_type!r}")
        if workflow_stage not in STAGES:
            raise ValueError(f"workflow_stage must be one of {sorted(STAGES)}")
        self.id = next_experiment_id()
        self.path = C.RUNS / self.id
        self.path.mkdir(parents=True)
        m = C.MEASUREMENT_METHOD
        self.record = {
            "experiment_id": self.id, "coursework_part": C.PART, "task": C.TASK,
            "run_type": run_type, "original_run_type": "", "activity": activity,
            "workflow_stage": workflow_stage, "model": model, "seed": seed,
            "training_examples": training_examples, "dataset_fraction": dataset_fraction,
            "examples_processed": 0, "epochs_completed": None, "selected_epoch": None,
            "stopping_point": "", "batch_size": batch_size, **hardware_info(),
            "carbon_intensity_gco2e_per_kwh": m["carbon_intensity_gco2e_per_kwh"],
            "carbon_intensity_type": m["carbon_intensity_type"],
            "carbon_intensity_source": m["carbon_intensity_source"],
            "f1_scale": "0-1", "validation_macro_f1": None, "test_macro_f1": None,
            "run_status": "running", "failure_reason": "",
            "cache_used": cache_used, "cache_path": cache_path,
            "checkpoint_used": checkpoint_used, "checkpoint_input_path": checkpoint_input_path,
            "evaluated_experiment_id": evaluated_experiment_id, "reuse_notes": reuse_notes,
            "rerun_of": rerun_of, "rerun_reason": rerun_reason,
            "trainable_parameters": None, "checkpoint_path": "", "predictions_path": "",
            "cache_output_path": "", "metrics": {}, "training_seconds": None,
            "measurement_method_id": m["measurement_method_id"],
            "tracker_name": m["tracker_name"], "tracker_version": _codecarbon_version(),
            "measurement_boundary": measurement_boundary, "missing_measurements": "",
            "notes": notes, "hyperparameters": hyperparameters or {},
            "measurement_method": m, "started_at": _now(), "finished_at": None,
        }
        self._early_stopped = False
        self._t_loop_start = self._t_loop_end = None
        self._save()  # registration before the run starts

    # -- public API ---------------------------------------------------------
    def update(self, **fields):
        unknown = set(fields) - set(self.record)
        if unknown:
            raise KeyError(f"Unknown run fields: {sorted(unknown)}")
        self.record.update(fields)
        self._save()

    def mark_training_start(self):
        """Time before this call is fixed setup cost (used in the scaling analysis)."""
        self._t_loop_start = time.perf_counter()

    def mark_training_end(self):
        self._t_loop_end = time.perf_counter()

    def early_stop(self, stopping_point):
        self._early_stopped = True
        self.record["stopping_point"] = stopping_point

    # -- context manager ----------------------------------------------------
    def __enter__(self):
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        from codecarbon import OfflineEmissionsTracker
        m = C.MEASUREMENT_METHOD
        self._t0 = time.perf_counter()
        self._sampler = _UtilSampler()
        self._sampler.start()
        self._tracker = OfflineEmissionsTracker(
            project_name=self.id, country_iso_code=m["country_iso_code"],
            measure_power_secs=m["measure_power_secs"], tracking_mode=m["tracking_mode"],
            pue=m["pue"], output_dir=str(self.path), save_to_file=True, log_level="error")
        self._tracker.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._tracker.stop()
        except Exception as err:  # never lose the record because the tracker failed
            self.record["missing_measurements"] += f"Tracker stop failed: {err}. "
        runtime = time.perf_counter() - self._t0
        util = self._sampler.stop()
        r = self.record
        r["runtime_seconds"] = round(runtime, 2)
        if self._t_loop_start is not None:
            end = self._t_loop_end or time.perf_counter()
            r["loop_seconds"] = round(end - self._t_loop_start, 2)
            r["setup_seconds"] = round(self._t_loop_start - self._t0, 2)
        r["cpu_utilisation_fraction"] = None if util is None else round(util, 3)
        try:
            self._fill_energy(runtime, util)
        except Exception as err:
            r["missing_measurements"] += f"Energy calculation failed: {err!r}. "

        if exc_type is None:
            r["run_status"] = "early-stopped" if self._early_stopped else "completed"
        else:
            r["run_status"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            r["original_run_type"] = r["run_type"]
            r["run_type"] = "failed"
            r["failure_reason"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            (self.path / "traceback.txt").write_text(
                "".join(traceback.format_exception(exc_type, exc, tb)), encoding="utf-8")
            if r["test_macro_f1"] is not None:
                r["test_macro_f1"] = None
        r["finished_at"] = _now()
        self._save()
        write_run_log()
        return False  # re-raise any exception

    # -- internals ----------------------------------------------------------
    def _fill_energy(self, runtime, util):
        m, r = C.MEASUREMENT_METHOD, self.record
        d = getattr(self._tracker, "final_emissions_data", None)
        cpu_mode = next((getattr(h, "_mode", None) for h in getattr(self._tracker, "_hardware", [])
                         if type(h).__name__ == "CPU"), None)
        if d is not None:
            r["energy_kwh"] = d.energy_consumed
            r["cpu_energy_kwh"], r["gpu_energy_kwh"], r["ram_energy_kwh"] = (
                d.cpu_energy, d.gpu_energy, d.ram_energy)
            r["cpu_power_w_avg"], r["ram_power_w_avg"] = d.cpu_power, d.ram_power
            r["tracker_duration_seconds"] = round(d.duration, 2)
        else:
            r["energy_kwh"] = None
            r["missing_measurements"] += "No CodeCarbon reading; see engineering estimate. "
        measured = cpu_mode in ("intel_rapl", "intel_power_gadget", "apple_powermetrics")
        r["cpu_energy_status"] = "measured" if measured else "estimated"
        r["gpu_energy_status"] = "none"
        r["ram_energy_status"] = "estimated"
        r["codecarbon_cpu_mode"] = cpu_mode
        if not measured:
            r["missing_measurements"] += (
                f"CPU power not metered (CodeCarbon mode '{cpu_mode}': constant 50% of TDP, "
                "independent of load). Utilisation-based estimate recorded as a cross-check. ")

        # Guidance 5.2: E = t * (P_cpu*u_cpu + P_gpu*u_gpu + P_ram) * PUE / 1000
        ram_w = m["ram_power_w_per_8gb"] * r["ram_gb"] / 8
        inputs = {"runtime_hours": runtime / 3600, "cpu_power_w": m["cpu_power_w"],
                  "cpu_utilisation": util, "gpu_power_w": 0.0, "gpu_utilisation": 0.0,
                  "ram_power_w": round(ram_w, 2), "pue": m["pue"]}
        if util is not None:
            r["engineering_estimate_kwh"] = (inputs["runtime_hours"] * (
                m["cpu_power_w"] * util + ram_w) * m["pue"] / 1000)
        r["engineering_estimate_inputs"] = inputs

        e = r["energy_kwh"] if r["energy_kwh"] is not None else r.get("engineering_estimate_kwh")
        if e is None:
            return
        if r["energy_kwh"] is None:
            r["energy_kwh"] = e
        ci = m["carbon_intensity_gco2e_per_kwh"]
        lo_ci, hi_ci = m["carbon_intensity_range_gco2e_per_kwh"]
        r["energy_kwh_low"] = e * m["energy_factor_low"]
        r["energy_kwh_high"] = e * m["energy_factor_high"] * m["unmeasured_overhead_high"]
        r["emissions_kgco2e"] = e * ci / 1000
        r["emissions_kgco2e_low"] = r["energy_kwh_low"] * lo_ci / 1000
        r["emissions_kgco2e_high"] = r["energy_kwh_high"] * hi_ci / 1000

    def _save(self):
        tmp = self.path / "run.json.tmp"
        tmp.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path / "run.json")


def _codecarbon_version():
    try:
        import codecarbon
        return codecarbon.__version__
    except Exception:
        return "unknown"


def load_runs():
    runs = []
    for p in sorted(C.RUNS.glob(f"{C.RUN_PREFIX}-*/run.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def write_run_log():
    """Rebuild logs/part_a/run_log.csv (the submitted experiment log) from every run.json."""
    C.LOGS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    with (C.LOGS / "run_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in LOG_COLUMNS})
    with (C.LOGS / "run_log.json").open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, default=str)
    return runs
