"""Experiment registry, hardware description and energy tracking for the environmental-impact log.

Each run lives in results/runs/<experiment_id>/run.json. It is registered (status "running") before any
work starts and updated when the run finishes, fails or is interrupted. results/run_log.csv is rebuilt
from all run.json files after every run, so failed runs are never lost.

Privacy: codecarbon records latitude/longitude. Only the whitelisted fields in ENERGY_FIELDS are copied;
no raw codecarbon output is written.
"""
import csv
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_TYPES = ("debug", "exploratory", "final", "failed")
ENERGY_FIELDS = ("duration", "energy_consumed", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
                 "cpu_power", "gpu_power", "ram_power")

LOG_COLUMNS = [
    "experiment_id", "member", "part", "task", "run_type", "status", "purpose", "rerun_reason",
    "model", "text_encoder", "imbalance", "seed", "train_examples", "dataset_fraction", "excluded_rows",
    "epochs_completed", "best_epoch", "stop_reason", "batch_size", "max_epochs", "patience", "lr",
    "trainable_params", "device", "cpu_model", "cpu_threads", "gpu_model", "gpu_count", "ram_gib",
    "runtime_s", "train_time_s", "eval_time_s", "energy_train_kwh", "energy_eval_kwh", "energy_kwh",
    "carbon_intensity_g_per_kwh", "carbon_intensity_source", "emissions_kgco2e", "energy_method",
    "tracker", "val_macro_f1", "test_macro_f1", "test_accuracy", "image_feature_cache", "checkpoint_reused",
    "started_at", "finished_at", "error",
]


def cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def hardware_info(device):
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        ram = None
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return {"device": str(device), "cpu_model": cpu_model(), "cpu_threads": os.cpu_count(),
            "gpu_model": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_count": torch.cuda.device_count() if cuda else 0, "ram_gib": ram,
            "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}


def energy_method(device):
    """Whether codecarbon can read real power counters on this machine, or falls back to estimates."""
    parts = []
    if platform.system() == "Linux" and os.access("/sys/class/powercap/intel-rapl:0/energy_uj", os.R_OK):
        parts.append("CPU measured (Intel RAPL)")
    else:
        parts.append("CPU estimated (codecarbon model, no RAPL access)")
    if str(device).startswith("cuda"):
        parts.append("GPU measured (NVML)")
    parts.append("RAM estimated")
    return "; ".join(parts)


class EnergyTracker:
    """Thin wrapper over codecarbon's offline tracker (no geolocation, nothing written to disk)."""

    def __init__(self, name, country="GBR", enabled=True):
        self.tracker, self.version = None, None
        if not enabled:
            return
        try:
            import codecarbon
            from codecarbon import OfflineEmissionsTracker
        except ImportError:
            print("codecarbon not installed: energy will not be recorded.", flush=True)
            return
        self.version = codecarbon.__version__
        self.tracker = OfflineEmissionsTracker(project_name=name, country_iso_code=country,
                                               log_level="error", save_to_file=False)

    def start(self):
        if self.tracker:
            self.tracker.start()

    def stop(self):
        """Return only the whitelisted energy fields (kWh, kg, W, s)."""
        if not self.tracker:
            return {}
        self.tracker.stop()
        data = getattr(self.tracker, "final_emissions_data", None)
        return {field: getattr(data, field, None) for field in ENERGY_FIELDS} if data else {}


def next_experiment_id(member, part, results_dir=RESULTS_DIR):
    """e.g. HEL-A-007: member prefix, coursework part, then a counter per member and part."""
    prefix = f"{member}-{part}-"
    runs = Path(results_dir) / "runs"
    numbers = [int(m.group(1)) for p in runs.glob(f"{prefix}*")
               if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", p.name))] if runs.exists() else []
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


REPO_ROOT = RESULTS_DIR.parent


def portable(path):
    """Path relative to the repo when inside it, so committed logs don't expose local home directories."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def save_run(run_dir, record, results_dir=RESULTS_DIR):
    write_json(Path(run_dir) / "run.json", record)
    rebuild_log(results_dir)


def rebuild_log(results_dir=RESULTS_DIR):
    """Flatten every run.json into results/run_log.csv (one row per run, sorted by experiment ID)."""
    records = [json.loads(p.read_text()) for p in sorted((Path(results_dir) / "runs").glob("*/run.json"))]
    with (Path(results_dir) / "run_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in LOG_COLUMNS})


def summarise_energy(train, evaluation, country):
    """Combine the training and evaluation phases into the log fields."""
    energy = sum(phase.get("energy_consumed") or 0 for phase in (train, evaluation))
    emissions = sum(phase.get("emissions") or 0 for phase in (train, evaluation))
    if not (train or evaluation):
        return {}
    return {"energy_train_kwh": train.get("energy_consumed"), "energy_eval_kwh": evaluation.get("energy_consumed"),
            "energy_kwh": energy, "emissions_kgco2e": emissions,
            "carbon_intensity_g_per_kwh": round(1000 * emissions / energy, 1) if energy else None,
            "carbon_intensity_source": f"codecarbon offline, country annual average ({country})",
            "energy_detail": {"train": train, "eval": evaluation}}
