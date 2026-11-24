"""Measure the idle machine for N seconds: the fixed background draw (scaling analysis).

    python -m part_a.idle_baseline --seconds 60

Close other programs first. On this laptop CodeCarbon uses a constant CPU power, so the
idle CodeCarbon reading equals the loaded one per second; the utilisation-based estimate
is what distinguishes idle from training. That is itself a finding worth reporting.
"""
import argparse
import time

from .energy import RunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    a = ap.parse_args()
    with RunLogger(run_type="final", workflow_stage="development", activity="idle_baseline",
                   model="none (idle machine)", batch_size=None,
                   measurement_boundary=f"Idle machine for {a.seconds} s with the tracker "
                   "running; no model code. Calibrates the fixed background power.") as run:
        run.update(trainable_parameters=0, examples_processed=0)
        time.sleep(a.seconds)
    print(f"[{run.id}] idle baseline: {run.record['energy_kwh']:.6f} kWh (CodeCarbon), "
          f"{run.record.get('engineering_estimate_kwh') or 0:.6f} kWh (utilisation estimate), "
          f"CPU util {run.record['cpu_utilisation_fraction']}")


if __name__ == "__main__":
    main()
