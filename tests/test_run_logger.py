"""Check logging behaviour without training models or measuring energy."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.aggregate_runs import aggregate
from scripts.run_logger import RunLogger, new_run_config

METHOD = json.loads(Path("templates/measurement_method.json").read_text())


def fake_hardware(cpu_mode):
    """Objects with the same class names/attributes CodeCarbon's hardware uses."""
    CPU = type("CPU", (), {})
    GPU = type("GPU", (), {})
    RAM = type("RAM", (), {})
    cpu, gpu, ram = CPU(), GPU(), RAM()
    cpu._mode = cpu_mode
    ram._force_ram_power = None
    return [ram, gpu, cpu]


def fake_tracker_class(cpu_mode="intel_rapl", cpu=0.004, gpu=0.005, ram=0.001, record=None):
    class FakeTracker:
        def __init__(self, **kwargs):
            if record is not None:
                record.update(kwargs)
            intensity = kwargs["force_carbon_intensity_g_co2e_kwh"]
            energy = cpu + gpu + ram
            self._hardware = fake_hardware(cpu_mode)
            self.final_emissions_data = SimpleNamespace(
                energy_consumed=energy, emissions=energy * intensity / 1000,
                cpu_energy=cpu, gpu_energy=gpu, ram_energy=ram,
                cpu_power=30.0, gpu_power=40.0, ram_power=20.0,
                cpu_utilization_percent=25.0, gpu_utilization_percent=50.0,
                cpu_model="test CPU", ram_total_size=8, duration=12.0,
            )

        def start(self):
            pass

        def stop(self):
            return self.final_emissions_data.emissions
    return FakeTracker


@patch("scripts.run_logger.torch.cuda.is_available", return_value=False)
class RunLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads(Path("templates/run_config.json").read_text())
        self.config["experiment_id"] = "logging_test"

    def read_summary(self, experiment_id="logging_test"):
        return json.loads((self.root / experiment_id / "summary.json").read_text())

    def test_epoch_curves_lr_selection_and_early_stop(self, _):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        with RunLogger(self.config, self.root, track_energy=False) as run:
            run.log_batch(0.8, optimizer, epoch=1, batch=1, examples=4)
            run.log_epoch(1, train_loss=0.8, valid_loss=0.9, valid_macro_f1=0.6, optimizer=optimizer)
            optimizer.param_groups[0]["lr"] = 0.05
            run.log_epoch(2, train_loss=0.7, valid_loss=0.8, valid_macro_f1=0.5, optimizer=optimizer)
            run.early_stop()
        summary = self.read_summary()
        self.assertEqual(summary["run_status"], "early-stopped")
        self.assertEqual(summary["run_type"], "final")
        self.assertEqual(summary["selected_epoch"], 1)
        self.assertEqual(summary["training_examples_processed"], 4)
        self.assertEqual(summary["f1_scale"], "0-1")
        self.assertIsNone(summary["energy_kwh"])
        self.assertIn("disabled", summary["missing_measurements"])
        with (run.path / "epochs.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([float(r["lr_group_0"]) for r in rows], [0.1, 0.05])
        events = EventAccumulator(str(run.path / "tensorboard")).Reload()
        self.assertEqual(len(events.Scalars("LR/epoch_group_0")), 2)
        self.assertAlmostEqual(events.Scalars("LR/epoch_group_0")[0].value, 0.1)
        self.assertAlmostEqual(events.Scalars("LR/epoch_group_0")[1].value, 0.05)
        self.assertIn("Loss/valid_epoch", events.Tags()["scalars"])

    def test_failure_reclassifies_run_type_and_preserves_record(self, _):
        with self.assertRaisesRegex(RuntimeError, "example failure"):
            with RunLogger(self.config, self.root, track_energy=False) as run:
                run.progress(epoch=1, batch=2, examples=3)
                raise RuntimeError("example failure")
        summary = self.read_summary()
        self.assertEqual(summary["run_status"], "failed")
        self.assertEqual(summary["run_type"], "failed")
        self.assertEqual(summary["original_run_type"], "final")
        self.assertEqual(summary["failure_reason"], "RuntimeError: example failure")
        self.assertEqual(summary["stopping_point"]["batch"], 2)
        with self.assertRaises(FileExistsError):
            with RunLogger(self.config, self.root, track_energy=False):
                pass

    def test_interruption_and_test_guard(self, _):
        self.config["run_type"] = "exploratory"
        with self.assertRaises(KeyboardInterrupt):
            with RunLogger(self.config, self.root, track_energy=False) as run:
                with self.assertRaises(ValueError):
                    run.log_test(0.8)
                raise KeyboardInterrupt()
        summary = self.read_summary()
        self.assertEqual(summary["run_status"], "interrupted")
        self.assertEqual(summary["run_type"], "failed")
        self.assertEqual(summary["original_run_type"], "exploratory")
        self.assertIsNone(summary["test_macro_f1"])

    def test_config_validation(self, _):
        for key, value in (("workflow_stage", "training"), ("run_type", "pilot"),
                           ("measurement_boundary", "")):
            config = dict(self.config, **{key: value})
            with self.assertRaises(ValueError, msg=key):
                RunLogger(config, self.root, track_energy=False)
        # Shared method values must not be silently overridden per run.
        with self.assertRaisesRegex(ValueError, "carbon_intensity_gco2e_per_kwh"):
            RunLogger(dict(self.config, carbon_intensity_gco2e_per_kwh=50), self.root, track_energy=False)

    def test_new_run_config(self, _):
        a = new_run_config("templates/run_config.json", "A_test", seed=1)
        b = new_run_config("templates/run_config.json", "A_test", seed=1)
        self.assertNotEqual(a["experiment_id"], b["experiment_id"])
        self.assertTrue(a["experiment_id"].startswith("A_test_"))
        self.assertEqual(a["seed"], 1)
        with self.assertRaises(KeyError):
            new_run_config("templates/run_config.json", "A_test", sede=1)

    def test_measured_energy_uncertainty_and_intensity(self, _):
        options = {}
        self.config["activity"] = "feature_extraction"
        with patch("codecarbon.OfflineEmissionsTracker", fake_tracker_class(record=options)):
            with RunLogger(self.config, self.root) as run:
                run.progress(split="valid", batch=1, examples=8)
        s = self.read_summary()
        self.assertEqual(options["force_carbon_intensity_g_co2e_kwh"], METHOD["carbon_intensity_gco2e_per_kwh"])
        self.assertEqual(options["pue"], METHOD["pue"])
        self.assertEqual(s["examples_processed"], 8)
        self.assertEqual(s["training_examples_processed"], 0)
        self.assertEqual((s["cpu_energy_status"], s["gpu_energy_status"], s["ram_energy_status"]),
                         ("measured", "measured", "estimated"))
        self.assertIn("CPU: measured (intel_rapl", s["measurement_method"])
        self.assertEqual(s["missing_measurements"], "")
        self.assertAlmostEqual(s["energy_kwh"], 0.010)
        self.assertAlmostEqual(s["emissions_kgco2e"], 0.010 * 122 / 1000)
        # low  = (0.004*0.9 + 0.005*0.95 + 0.001*0.25) * 1.0
        # high = (0.004*1.1 + 0.005*1.05 + 0.001*1.0) * 1.25
        self.assertAlmostEqual(s["energy_kwh_low"], 0.0086)
        self.assertAlmostEqual(s["energy_kwh_high"], 0.0133125)
        self.assertAlmostEqual(s["emissions_kgco2e_low"], 0.0086 * 94 / 1000)
        self.assertAlmostEqual(s["emissions_kgco2e_high"], 0.0133125 * 237.589 / 1000)
        self.assertLess(s["energy_kwh_low"], s["energy_kwh"])
        self.assertGreater(s["energy_kwh_high"], s["energy_kwh"])
        self.assertEqual(s["tracker_duration_seconds"], 12.0)
        self.assertEqual(s["gpu_utilisation_fraction"], 0.5)
        self.assertEqual(s["carbon_intensity_type"], "national_12_month_average")

    def test_estimated_cpu_widens_range_and_is_declared(self, _):
        with patch("codecarbon.OfflineEmissionsTracker", fake_tracker_class(cpu_mode="cpu_load")):
            with RunLogger(self.config, self.root):
                pass
        s = self.read_summary()
        self.assertEqual(s["cpu_energy_status"], "estimated")
        self.assertIn("CPU energy is estimated", s["missing_measurements"])
        self.assertEqual(s["uncertainty_factors_applied"]["cpu"]["factor_key"], "cpu_estimated")
        self.assertAlmostEqual(s["energy_kwh_high"], (0.004 * 2.0 + 0.005 * 1.05 + 0.001) * 1.25)

    def test_evaluation_must_link_to_training_run(self, _):
        evaluation = json.loads(Path("templates/evaluation_config.json").read_text())
        evaluation["experiment_id"] = "eval_test"
        with self.assertRaisesRegex(ValueError, "evaluated_experiment_id"):
            RunLogger(evaluation, self.root, track_energy=False)
        evaluation["evaluated_experiment_id"] = "logging_test"
        with self.assertRaisesRegex(ValueError, "no record"):
            RunLogger(evaluation, self.root, track_energy=False)
        with RunLogger(self.config, self.root, track_energy=False):
            pass
        with RunLogger(evaluation, self.root, track_energy=False) as run:
            run.log_test(0.75)
        self.assertEqual(self.read_summary("eval_test")["test_macro_f1"], 0.75)

    def test_aggregate_tables(self, _):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        evaluation = json.loads(Path("templates/evaluation_config.json").read_text())
        for seed, (val, test, cpu) in enumerate([(0.70, 0.68, 0.004), (0.72, 0.70, 0.006)]):
            train_id, eval_id = f"train_s{seed}", f"eval_s{seed}"
            with patch("codecarbon.OfflineEmissionsTracker", fake_tracker_class(cpu=cpu)):
                with RunLogger(dict(self.config, experiment_id=train_id, seed=seed), self.root) as run:
                    run.log_epoch(1, train_loss=1, valid_loss=1, valid_macro_f1=val, optimizer=optimizer)
                with RunLogger(dict(evaluation, experiment_id=eval_id, seed=seed,
                                    evaluated_experiment_id=train_id), self.root) as run:
                    run.log_test(test)
        with self.assertRaises(RuntimeError):
            with RunLogger(dict(self.config, experiment_id="crash", run_type="exploratory",
                                workflow_stage="development"), self.root, track_energy=False):
                raise RuntimeError("bug")

        out = self.root / "out"
        summary = aggregate(self.root, out)
        self.assertEqual(summary["runs"], 5)
        self.assertEqual(summary["runs_by_run_type"], {"debug": 0, "exploratory": 0, "final": 4, "failed": 1})
        self.assertTrue(any("crash" in w for w in summary["warnings"]))
        with (out / "run_log.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows[0])[:4], ["experiment_id", "coursework_part", "task", "run_type"])
        with (out / "stage_summary.csv").open() as f:
            stages = {r["stage"]: r for r in csv.DictReader(f)}
        # 2 training runs (0.010 + 0.012) and 2 evaluations (0.010 + 0.012); each counted once.
        self.assertAlmostEqual(float(stages["final_training"]["energy_kwh"]), 0.022)
        self.assertAlmostEqual(float(stages["evaluation"]["energy_kwh"]), 0.022)
        self.assertAlmostEqual(float(stages["TOTAL"]["energy_kwh"]), 0.044)
        with (out / "model_summary.csv").open() as f:
            models = list(csv.DictReader(f))
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["seeds"], "[0, 1]")
        self.assertAlmostEqual(float(models[0]["test_macro_f1_mean"]), 0.69)
        self.assertAlmostEqual(float(models[0]["energy_kwh_total"]), 0.044)
        self.assertAlmostEqual(float(models[0]["test_f1_per_kwh"]), 0.69 / 0.022)

    def test_live_energy_events_and_interval_power(self, _):
        from torch.utils.tensorboard import SummaryWriter
        from scripts.energy_tensorboard import TensorBoardEnergyOutput

        writer = SummaryWriter(str(self.root / "events"))
        self.addCleanup(writer.close)
        output = TensorBoardEnergyOutput(writer, self.root / "energy_timeseries.csv")
        def sample(duration, scale):
            return SimpleNamespace(duration=duration, energy_consumed=0.01*scale,
                                   cpu_energy=0.006*scale, gpu_energy=0.003*scale,
                                   ram_energy=0.001*scale, emissions=0.002*scale,
                                   cpu_power=2160, gpu_power=1080, ram_power=360)

        output.live_out(sample(10, 1), None)
        output.live_out(sample(20, 1.5), None)
        # Verify events while the writer/run is still open, not only at exit.
        events = EventAccumulator(str(self.root / "events")).Reload()
        self.assertEqual(len(events.Scalars("Energy/total_kwh")), 2)
        self.assertAlmostEqual(events.Scalars("Power/total_w_interval")[0].value, 3600)
        self.assertAlmostEqual(events.Scalars("Power/total_w_interval")[1].value, 1800)
        self.assertAlmostEqual(events.Scalars("Energy/gpu_kwh")[1].value, 0.0045)
        output.out(sample(20, 1.5), None)  # Duplicate final callback is ignored.
        with output.csv_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[-1]["energy_total_kwh"]), 0.015)
        self.assertIsNone(output.error)

    def test_notebook_feature_function_and_cache_alignment(self, _):
        import ast

        notebook = json.loads(Path("PartA.ipynb").read_text())
        functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                functions += [node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == "cache_features"]
        self.assertEqual(len(functions), 1)

        class SmallDataset(torch.utils.data.Dataset):
            label_to_id = {"not_informative": 0, "informative": 1}
            def __len__(self):
                return 3
            def __getitem__(self, i):
                return dict(image=torch.full((3, 2, 2), float(i)), label=i % 2,
                            image_id=f"large_id_{i}", tweet_id=f"tweet_{i}")

        class SmallEncoder(torch.nn.Module):
            def forward(self, images):
                return images.mean((1, 2, 3))[:, None].expand(-1, 512)

        cache_dir = self.root / "cache"
        cache_dir.mkdir()
        namespace = dict(torch=torch, DataLoader=torch.utils.data.DataLoader,
                         device="cpu", cache_dir=cache_dir,
                         weights=SimpleNamespace(name="TEST_WEIGHTS"))
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook_cache", "exec"), namespace)
        config = json.loads(Path("templates/feature_extraction_config.json").read_text())
        config.update(experiment_id="logging_test", batch_size=2)
        with RunLogger(config, self.root, track_energy=False) as run:
            path = namespace["cache_features"](SmallDataset(), "train", run, SmallEncoder())
            run.update(cache_output_paths=[path])
        cache = torch.load(path, weights_only=True)
        self.assertEqual(cache["features"].shape, (3, 512))
        self.assertEqual(cache["image_ids"], [f"large_id_{i}" for i in range(3)])
        self.assertEqual(cache["features"][:, 0].tolist(), [0., 1., 2.])
        summary = self.read_summary()
        self.assertEqual(summary["examples_processed"], 3)
        self.assertEqual(summary["workflow_stage"], "artefact_production")
        self.assertEqual(summary["epochs_completed"], 0)
        self.assertIsNone(summary["validation_macro_f1"])
        self.assertEqual(summary["cache_output_paths"], [path])


if __name__ == "__main__":
    unittest.main()
