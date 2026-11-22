"""Glue between our training code and the group's run logger (scripts/run_logger.py, kept unchanged).

The logger reads `templates/measurement_method.json` and imports `scripts.*` relative to the repository
root, so scripts call `enter_repo_root()` first. Records go to logs/runs/<experiment_id>/.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = "logs/runs"  # relative, so stored paths never contain a home directory
TASK_NAMES = {"informative": "binary_informativeness", "humanitarian": "five_class_humanitarian"}
MACHINE_NOTE = ("CodeCarbon machine mode measures the whole computer, so the idle desktop, display output "
                "and any other open programs are included; see the idle-baseline run for this fixed "
                "background draw.")
TRAIN_BOUNDARY = ("Loading the prepared TSVs and cached image features, tokenisation and training-only "
                  "vocabulary, model construction, training, per-epoch validation, best-epoch validation "
                  "predictions and checkpoint saving; excludes feature extraction (own record), dataset "
                  "download and test evaluation (separate evaluation record). " + MACHINE_NOTE)
EVAL_BOUNDARY = ("Loading the selected checkpoint, the test TSV and cached image features, tokenisation with "
                 "the checkpoint's training vocabulary, test-set prediction, metric calculation and saving "
                 "predictions; excludes training (see evaluated_experiment_id). " + MACHINE_NOTE)


def enter_repo_root():
    """Make the logger's relative paths and `scripts.*` imports work from any working directory."""
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def relative(path):
    """Repo-relative POSIX path (absolute paths would expose the local home directory in logs)."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.name


def model_key(model, text_encoder=None, imbalance="none"):
    """Short configuration name, e.g. 'fusion_textcnn' or 'fusion_textcnn_weighted_loss'."""
    parts = [model] + ([text_encoder] if model in ("text", "fusion") and text_encoder else [])
    if imbalance and imbalance != "none":
        parts.append(imbalance)
    return "_".join(parts)


def workflow_stage(part, run_type):
    if str(part).upper().startswith("C"):
        return "part_c_extension"
    return "final_training" if run_type == "final" else "development"
