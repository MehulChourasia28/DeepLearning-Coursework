"""CodeCarbon output adapter: stream sampled energy into TensorBoard and CSV."""
import csv
import math
import threading

from codecarbon.output_methods.base_output import BaseOutput


class TensorBoardEnergyOutput(BaseOutput):
    """Use public CodeCarbon callbacks; do not start a second energy tracker.

    Energy is cumulative. Interval power is derived from the change in energy
    between callbacks, not an instantaneous sensor reading. CodeCarbon's
    cpu_power/gpu_power/ram_power values are averages over the whole run.
    """
    def __init__(self, writer, csv_path):
        self.writer = writer
        self.csv_path = csv_path
        self.previous = None
        self.sample = 0
        self.lock = threading.Lock()
        self.error = None

    def live_out(self, total, delta):
        self._record(total)

    def out(self, total, delta):
        self._record(total)

    def _record(self, data):
        # A dashboard failure must not stop the actual feature extraction.
        try:
            with self.lock:
                self._write(data)
        except Exception as exc:
            self.error = f"TensorBoard energy output: {type(exc).__name__}: {exc}"

    def _write(self, data):
        fields = {
            "elapsed_seconds": ("Runtime/elapsed_seconds", "duration"),
            "energy_total_kwh": ("Energy/total_kwh", "energy_consumed"),
            "energy_cpu_kwh": ("Energy/cpu_kwh", "cpu_energy"),
            "energy_gpu_kwh": ("Energy/gpu_kwh", "gpu_energy"),
            "energy_ram_kwh": ("Energy/ram_kwh", "ram_energy"),
            "emissions_kgco2e": ("Emissions/cumulative_kgco2e", "emissions"),
            "cpu_w_run_average": ("Power/cpu_w_run_average", "cpu_power"),
            "gpu_w_run_average": ("Power/gpu_w_run_average", "gpu_power"),
            "ram_w_run_average": ("Power/ram_w_run_average", "ram_power"),
        }
        row = {key: getattr(data, attr, None) for key, (_, attr) in fields.items()}
        row = {key: float(value) if value is not None and math.isfinite(float(value)) else None
               for key, value in row.items()}
        if row["elapsed_seconds"] is None:
            return
        if self.previous and row["elapsed_seconds"] <= self.previous["elapsed_seconds"]:
            return
        self.sample += 1
        for key, (tag, _) in fields.items():
            if row[key] is not None:
                self.writer.add_scalar(tag, row[key], self.sample)

        previous = self.previous or {key: 0.0 for key in row}
        seconds = row["elapsed_seconds"] - previous["elapsed_seconds"]
        for component in ("total", "cpu", "gpu", "ram"):
            key = f"energy_{component}_kwh"
            power = None
            if seconds > 0 and row[key] is not None and previous.get(key) is not None:
                # kWh / seconds -> W. Tiny negative sensor rounding is clipped.
                power = max(0.0, row[key] - previous[key]) * 3_600_000 / seconds
                self.writer.add_scalar(f"Power/{component}_w_interval", power, self.sample)
            row[f"{component}_w_interval"] = power

        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample", *row])
            if not exists:
                writer.writeheader()
            writer.writerow({"sample": self.sample, **row})
        self.previous = row
        self.writer.flush()
