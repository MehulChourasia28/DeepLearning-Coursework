"""Per-run environmental records, epoch CSVs and TensorBoard scalars.

Use as a context manager. Exceptions are recorded and then propagated.
Each experiment ID must be new: retries/resumed attempts get a new ID.

Shared measurement assumptions (carbon intensity, PUE, uncertainty factors)
come from templates/measurement_method.json and are copied into every run
record, so each summary.json is self-contained.
"""
import csv
import json
import logging
import math
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

RUN_TYPES = {"debug", "exploratory", "final", "failed"}
WORKFLOW_STAGES = {"development", "final_training", "evaluation",
                   "part_c_extension", "artefact_production"}
REQUIRED_CONFIG = ("experiment_id", "coursework_part", "task", "run_type", "model",
                   "activity", "workflow_stage", "measurement_boundary")
DEFAULT_METHOD_FILE = "templates/measurement_method.json"
# CodeCarbon CPU modes that read a hardware energy counter rather than a model.
MEASURED_CPU_MODES = {"intel_rapl", "intel_power_gadget", "windows_emi"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _clean(value):
    """JSON cannot hold NaN/inf; store them as null rather than losing the record."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def new_run_config(template_path, id_prefix, **overrides):
    """Load a template, apply overrides and assign a fresh unique experiment ID.

    Overrides must already exist in the template, so a typo cannot silently
    add an unused field.
    """
    config = json.loads(Path(template_path).read_text())
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise KeyError(f"Not in template {template_path}: {unknown}")
    config.update(overrides)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    config["experiment_id"] = f"{id_prefix}_{stamp}_{uuid4().hex[:8]}"
    return config


class RunLogger:
    def __init__(self, config, root="logs/runs", track_energy=True):
        config = dict(config)
        method_file = Path(config.get("measurement_method_file") or DEFAULT_METHOD_FILE)
        method = json.loads(method_file.read_text())
        clashes = sorted(k for k in method if config.get(k) is not None and config[k] != method[k])
        if clashes:
            raise ValueError(f"Set these in {method_file}, not the run config: {clashes}")
        self.config = {**config, **method, "measurement_method_file": str(method_file)}

        for key in REQUIRED_CONFIG:
            if self.config.get(key) in (None, ""):
                raise ValueError(f"Missing configuration: {key}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.config["experiment_id"]) or self.config["experiment_id"] in {".", ".."}:
            raise ValueError("experiment_id must be a simple filename, without spaces or slashes")
        if self.config["run_type"] not in RUN_TYPES:
            raise ValueError(f"Invalid run_type; use one of {sorted(RUN_TYPES)}")
        if self.config["workflow_stage"] not in WORKFLOW_STAGES:
            raise ValueError(f"Invalid workflow_stage; use one of {sorted(WORKFLOW_STAGES)}")
        if self.config["activity"] == "evaluation":
            evaluated = self.config.get("evaluated_experiment_id")
            if not evaluated:
                raise ValueError("Evaluation runs must set evaluated_experiment_id")
            if not (Path(root) / evaluated / "summary.json").exists():
                raise ValueError(f"evaluated_experiment_id {evaluated!r} has no record in {root}")
            if not self.config.get("checkpoint_used"):
                raise ValueError("Evaluation runs must set checkpoint_used=true")

        self.path = Path(root) / self.config["experiment_id"]
        self.track_energy = track_energy
        self.tracker = None
        self.handler = None
        self.writer = None
        self.energy_output = None
        self.best_f1 = None
        self.last_epoch = 0
        self.step = 0
        cuda = torch.cuda.is_available()
        self.summary = {
            **self.config, "run_status": "registered", "original_run_type": None,
            "failure_reason": None, "started_at": None, "finished_at": None,
            "epochs_completed": 0, "selected_epoch": None, "stopping_point": None,
            "runtime_seconds": None, "tracker_duration_seconds": None,
            "energy_kwh": None, "energy_kwh_low": None, "energy_kwh_high": None,
            "emissions_kgco2e": None, "emissions_kgco2e_low": None,
            "emissions_kgco2e_high": None, "uncertainty_factors_applied": None,
            "cpu_energy_kwh": None, "gpu_energy_kwh": None, "ram_energy_kwh": None,
            "cpu_energy_status": None, "gpu_energy_status": None, "ram_energy_status": None,
            "measurement_method": None,
            "cpu_power_w_avg": None, "gpu_power_w_avg": None, "ram_power_w_avg": None,
            "cpu_utilisation_fraction": None, "gpu_utilisation_fraction": None,
            "f1_scale": "0-1", "validation_macro_f1": None, "test_macro_f1": None,
            "checkpoint_path": None, "predictions_path": None,
            "training_examples_processed": 0, "examples_processed": 0,
            "execution_device": "cuda" if cuda else "cpu",
            "cpu_model": _cpu_name(),
            "gpu_model": torch.cuda.get_device_name(0) if cuda else "N/A",
            "device_count": torch.cuda.device_count() if cuda else 1,
            "ram_gb": psutil.virtual_memory().total / 1024**3,
            "os": platform.platform(),
            "tracker_name": "CodeCarbon" if track_energy else "disabled",
            "tracker_version": version("codecarbon") if track_energy else None,
            "torch_version": str(torch.__version__), "tensorboard_version": version("tensorboard"),
            "trainable_parameters": None,
            "cache_output_paths": [],
            "missing_measurements": self.config.get("missing_measurements") or "",
        }

    def _persist(self):
        _write_json(self.path / "summary.json", self.summary)

    def _note_missing(self, text):
        self.summary["missing_measurements"] = (
            (self.summary.get("missing_measurements") or "") + "; " + text
        ).lstrip("; ")

    def __enter__(self):
        # Refuse to overwrite a previous experiment, even after a failed run.
        self.path.mkdir(parents=True, exist_ok=False)
        _write_json(self.path / "config.json", self.config)
        self._persist()
        self.writer = SummaryWriter(str(self.path / "tensorboard"), flush_secs=5)
        self.writer.add_text("Run/config", json.dumps(_clean(self.config), indent=2), 0)
        self.started = time.perf_counter()
        self.summary.update(started_at=_now(), run_status="running")
        self._persist()
        try:
            if self.track_energy:
                from codecarbon import OfflineEmissionsTracker
                from scripts.energy_tensorboard import TensorBoardEnergyOutput
                self.energy_output = TensorBoardEnergyOutput(
                    self.writer, self.path / "energy_timeseries.csv"
                )
                self.handler = logging.FileHandler(self.path / "codecarbon.log")
                logging.getLogger("codecarbon").addHandler(self.handler)
                self.tracker = OfflineEmissionsTracker(
                    country_iso_code=self.config.get("country_iso_code", "GBR"),
                    project_name=self.config["experiment_id"],
                    tracking_mode=self.config.get("tracking_mode", "machine"),
                    measure_power_secs=self.config.get("measure_power_secs", 5),
                    api_call_interval=1,
                    output_handlers=[self.energy_output],
                    output_dir=str(self.path), output_file="energy.csv", log_level="info",
                    force_carbon_intensity_g_co2e_kwh=self.config["carbon_intensity_gco2e_per_kwh"],
                    pue=self.config["pue"],
                )
                # Keep full INFO detail in codecarbon.log but stop per-second console spam.
                # CodeCarbon recreates its console handler inside the tracker constructor.
                console_level = getattr(logging, str(self.config.get("console_log_level", "ERROR")).upper())
                for handler in logging.getLogger("codecarbon").handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        handler.setLevel(console_level)
                self._synchronize()
                self.tracker.start()
                self._record_component_methods()
                self._persist()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def _record_component_methods(self):
        """Record, per component, whether energy is measured or estimated.

        CodeCarbon caches hardware setup per process, so its setup log is only
        written for the first tracker; the hardware objects are always present.
        """
        status = {"cpu": ("none", "not tracked"), "gpu": ("none", "no GPU tracked"),
                  "ram": ("none", "not tracked")}
        try:
            for hardware in self.tracker._hardware:
                name = type(hardware).__name__
                if name == "CPU":
                    mode = getattr(hardware, "_mode", None)
                    if mode in MEASURED_CPU_MODES:
                        status["cpu"] = ("measured", f"{mode} package energy counter")
                    else:
                        status["cpu"] = ("estimated", f"CodeCarbon '{mode}' model from TDP and CPU load; no readable energy counter")
                elif name == "GPU":
                    status["gpu"] = ("measured", "GPU driver energy counter (pynvml/amdsmi)")
                elif name == "RAM":
                    forced = getattr(hardware, "_force_ram_power", None)
                    status["ram"] = ("estimated", f"user constant {forced} W" if forced is not None
                                     else "CodeCarbon heuristic, 5 W per estimated DIMM")
                elif name == "AppleSiliconChip":
                    part = getattr(hardware, "chip_part", "CPU").lower()
                    status["gpu" if part == "gpu" else "cpu"] = ("measured", "macOS powermetrics")
                else:
                    self._note_missing(f"Unrecognised CodeCarbon hardware {name}; treated as estimated")
        except Exception as exc:
            self._note_missing(f"Could not inspect tracker hardware: {type(exc).__name__}: {exc}")
        for component, (kind, how) in status.items():
            self.summary[f"{component}_energy_status"] = kind
        self.summary["measurement_method"] = (
            f"CodeCarbon {self.summary['tracker_version']} OfflineEmissionsTracker, "
            f"tracking_mode={self.config.get('tracking_mode', 'machine')}; "
            + "; ".join(f"{c.upper()}: {k} ({h})" for c, (k, h) in status.items())
        )
        if status["cpu"][0] == "estimated":
            self._note_missing("CPU energy is estimated, not measured: RAPL energy counter unreadable "
                               "(run: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj)")

    @staticmethod
    def _synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def update(self, **fields):
        """Record manual fields, including outputs and method notes."""
        for key in fields:
            if key not in self.summary:
                raise KeyError(f"Unknown summary field: {key}")
        self.summary.update(fields)
        self._persist()

    def progress(self, *, epoch=None, batch=None, examples=0, split="train"):
        """Call after each successful batch; counters survive normal interruptions.

        Training batches count as training examples; everything else (feature
        extraction, debug forward passes, evaluation) counts as examples_processed.
        """
        if examples < 0:
            raise ValueError("examples must be non-negative")
        self.step += 1
        if self.config["activity"] == "training" and split == "train":
            self.summary["training_examples_processed"] += examples
        else:
            self.summary["examples_processed"] += examples
        self.summary["stopping_point"] = {"split": split, "epoch": epoch, "batch": batch}
        self._persist()

    def log_batch(self, loss, optimizer, *, epoch=None, batch=None, examples=0):
        self.progress(epoch=epoch, batch=batch, examples=examples)
        self.writer.add_scalar("Loss/train_batch", float(loss), self.step)
        for i, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"LR/batch_group_{i}", group["lr"], self.step)

    def log_epoch(self, epoch, *, train_loss, valid_loss, valid_macro_f1, optimizer):
        """Log AFTER validation but BEFORE an epoch-level scheduler.step()."""
        if epoch <= self.last_epoch:
            raise ValueError("Epoch numbers must strictly increase, starting at 1")
        values = [float(train_loss), float(valid_loss), float(valid_macro_f1)]
        if not all(math.isfinite(x) for x in values) or not 0 <= values[2] <= 1:
            raise ValueError("Metrics must be finite and F1 must be in [0, 1]")
        row = dict(epoch=epoch, train_loss=values[0], valid_loss=values[1],
                   valid_macro_f1=values[2], elapsed_seconds=time.perf_counter()-self.started)
        for i, group in enumerate(optimizer.param_groups):
            row[f"lr_group_{i}"] = group["lr"]
            self.writer.add_scalar(f"LR/epoch_group_{i}", group["lr"], epoch)
        path = self.path / "epochs.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        for tag, value in (("Loss/train_epoch", values[0]), ("Loss/valid_epoch", values[1]),
                           ("F1/valid_macro", values[2])):
            self.writer.add_scalar(tag, value, epoch)
        if self.best_f1 is None or values[2] > self.best_f1:
            self.best_f1 = values[2]
            self.summary.update(validation_macro_f1=values[2], selected_epoch=epoch)
        self.last_epoch = epoch
        self.summary["epochs_completed"] = epoch
        self._persist()
        self.writer.flush()

    def log_test(self, macro_f1):
        """Only for the final evaluation of the already selected checkpoint."""
        if self.config["run_type"] != "final":
            raise ValueError("Test metrics are restricted to final evaluation runs")
        value = float(macro_f1)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Test macro-F1 must be in [0, 1]")
        self.summary["test_macro_f1"] = value
        self.writer.add_scalar("F1/test_final", value, self.last_epoch)
        self._persist()

    def early_stop(self):
        self.summary["run_status"] = "early-stopped"
        self._persist()

    def describe(self):
        """Human-readable run report for notebook output."""
        s = self.summary

        def f(key, spec):
            return "n/a" if s.get(key) is None else format(s[key], spec)

        lines = [
            f"{s['experiment_id']}: {s['run_status']} (run_type={s['run_type']}, stage={s['workflow_stage']})",
            f"  logs: {self.path}",
            f"  runtime {f('runtime_seconds', '.1f')} s (tracker {f('tracker_duration_seconds', '.1f')} s)",
            f"  energy {f('energy_kwh', '.6f')} kWh, range {f('energy_kwh_low', '.6f')} to {f('energy_kwh_high', '.6f')}",
            f"  emissions {f('emissions_kgco2e', '.6f')} kgCO2e, range {f('emissions_kgco2e_low', '.6f')} "
            f"to {f('emissions_kgco2e_high', '.6f')} (central {s['carbon_intensity_gco2e_per_kwh']} gCO2/kWh)",
            f"  average power: CPU {f('cpu_power_w_avg', '.1f')} W, GPU {f('gpu_power_w_avg', '.1f')} W, "
            f"RAM {f('ram_power_w_avg', '.1f')} W",
            f"  method: {s['measurement_method']}",
        ]
        if s.get("failure_reason"):
            lines.append(f"  failure: {s['failure_reason']}")
        if s.get("missing_measurements"):
            lines.append(f"  missing: {s['missing_measurements']}")
        return "\n".join(lines)

    def _record_energy(self, data):
        energy = _float_or_none(data.energy_consumed)
        emissions = _float_or_none(data.emissions)
        components = {c: _float_or_none(getattr(data, f"{c}_energy", None)) for c in ("cpu", "gpu", "ram")}
        self.summary.update(
            energy_kwh=energy, emissions_kgco2e=emissions,
            cpu_model=getattr(data, "cpu_model", None) or self.summary["cpu_model"],
            ram_gb=_float_or_none(getattr(data, "ram_total_size", None)) or self.summary["ram_gb"],
            tracker_duration_seconds=_float_or_none(getattr(data, "duration", None)),
            **{f"{c}_energy_kwh": e for c, e in components.items()},
        )
        for c in ("cpu", "gpu", "ram"):
            self.summary[f"{c}_power_w_avg"] = _float_or_none(getattr(data, f"{c}_power", None))
        for c in ("cpu", "gpu"):
            percent = _float_or_none(getattr(data, f"{c}_utilization_percent", None))
            self.summary[f"{c}_utilisation_fraction"] = None if percent is None else percent / 100
        if energy is not None:
            self.writer.add_scalar("Energy/kwh_total", energy, self.last_epoch)
        if emissions is not None:
            self.writer.add_scalar("Emissions/kgco2e_total", emissions, self.last_epoch)
        self._record_uncertainty(components)

    def _record_uncertainty(self, components):
        """Low/high bounds from per-component factors and the intensity range."""
        if any(e is None for e in components.values()):
            self._note_missing("Component energy missing; uncertainty bounds not computed")
            return
        factors = self.config["component_energy_factors"]
        applied, low, high = {}, 0.0, 0.0
        for component, energy in components.items():
            status = self.summary.get(f"{component}_energy_status") or "unknown"
            key = "none" if status == "none" else f"{component}_{status}"
            if key not in factors:
                key = "unknown"
            lo, hi = factors.get(key, [0.5, 2.0])
            applied[component] = {"factor_key": key, "low": lo, "high": hi}
            low += energy * lo
            high += energy * hi
        overhead_lo, overhead_hi = self.config["unmeasured_overhead_factor"]
        ci_lo, ci_hi = self.config["carbon_intensity_range_gco2e_per_kwh"]
        applied["unmeasured_overhead"] = {"low": overhead_lo, "high": overhead_hi}
        applied["carbon_intensity_gco2e_per_kwh"] = {"low": ci_lo, "high": ci_hi}
        low, high = low * overhead_lo, high * overhead_hi
        self.summary.update(
            energy_kwh_low=low, energy_kwh_high=high,
            emissions_kgco2e_low=low * ci_lo / 1000, emissions_kgco2e_high=high * ci_hi / 1000,
            uncertainty_factors_applied=applied,
        )

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.summary["run_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.summary["failure_reason"] = f"{exc_type.__name__}: {exc}"
            # Guidance: failed runs are classified as run_type "failed"; keep the purpose.
            if self.summary["run_type"] != "failed":
                self.summary["original_run_type"] = self.summary["run_type"]
                self.summary["run_type"] = "failed"
        elif self.summary["run_status"] == "running":
            self.summary["run_status"] = "completed"
        try:
            try:
                self._synchronize()
            except Exception as sync_error:
                self._note_missing(f"GPU synchronization: {sync_error}")
            if self.tracker is not None:
                self.tracker.stop()
                data = self.tracker.final_emissions_data
                if data is not None:
                    self._record_energy(data)
                else:
                    self._note_missing("Tracker returned no final energy data")
        except Exception as measurement_error:
            self._note_missing(f"Tracker finalization: {measurement_error}")
        finally:
            self.summary.update(finished_at=_now(), runtime_seconds=time.perf_counter()-self.started)
            if not self.track_energy:
                self._note_missing("Energy tracking explicitly disabled; energy/emissions are null")
            if self.energy_output is not None and self.energy_output.error:
                self._note_missing(self.energy_output.error)
            self._persist()
            if self.writer is not None:
                self.writer.close()
            if self.handler is not None:
                logging.getLogger("codecarbon").removeHandler(self.handler)
                self.handler.close()
        return False
