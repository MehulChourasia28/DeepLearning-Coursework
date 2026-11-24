"""Paths, hyperparameters from the coursework brief, and the energy measurement method."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crisismmd"
PREPARED = DATA / "prepared" / "informative"
FEATURE_CACHE = DATA / "features" / "resnet18_imagenet1k_v1.npz"

# Short paths on purpose: the repo lives deep inside OneDrive and Windows has a
# 260-character path limit.
# Helithas Linux file paths messed up my github initially 
OUT = ROOT / "results" / "part_a"
RUNS = OUT / "runs"
FIGURES = OUT / "figures"
LOGS = ROOT / "logs" / "part_a"

RUN_PREFIX = "AYN-A"          # experiment IDs: AYN-A-001, AYN-A-002, ...
PART = "A"
TASK = "binary_informativeness"
LABELS = ["informative", "not_informative"]   # index 0 = informative (positive class)
POSITIVE = "informative"

# Coursework brief: 128-d embeddings, 64-token limit, Adam, batch 64, <=20 epochs,
# early-stopping patience 3 on validation macro-F1, seeds 0/1/2.
HPARAMS = {
    "max_tokens": 64,
    "embed_dim": 128,
    "min_freq": 2,              # training-only vocabulary; words seen once -> <unk>
    "textcnn_kernels": [2, 3, 4, 5],
    "textcnn_filters": 64,      # 4 x 64 = 256-d text feature
    "lstm_hidden": 128,         # 2 directions x 128 = 256-d text feature (same as TextCNN)
    "image_dim": 512,
    "head_hidden": 256,
    "dropout": 0.5,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 20,
    "patience": 3,
}
SEEDS = [0, 1, 2]
MODELS = ["text_textcnn", "text_bilstm", "image_mlp", "fusion_textcnn", "fusion_bilstm"]

# ---------------------------------------------------------------------------
# Measurement method (copied into every run record). Values match the group's
# shared assumptions so logs from all members can be aggregated together.
# ---------------------------------------------------------------------------
MEASUREMENT_METHOD = {
    "measurement_method_id": "M1-AYN_codecarbon282_constant_plus_util_estimate",
    "tracker_name": "CodeCarbon OfflineEmissionsTracker",
    "tracking_mode": "machine",
    "measure_power_secs": 5,
    "country_iso_code": "GBR",
    "carbon_intensity_gco2e_per_kwh": 122,
    "carbon_intensity_type": "national_12_month_average",
    "carbon_intensity_source": (
        "NESO Carbon Intensity API (api.carbonintensity.org.uk), Great Britain national "
        "intensity, mean of monthly averages Nov 2025 - Oct 2026 = 121.7, rounded to 122 "
        "gCO2/kWh. Annual national average, not time-specific or regional. Same value as the "
        "group's shared measurement method."),
    "carbon_intensity_range_gco2e_per_kwh": [94, 237.589],
    "carbon_intensity_range_source": (
        "Low: lowest NESO monthly national average in the same 12 months (Apr 2026). "
        "High: CodeCarbon bundled GBR annual intensity (237.589 g/kWh)."),
    "pue": 1.0,
    "pue_source": "Personal laptop, no data-centre cooling overhead.",
    # Engineering estimate (guidance section 5.2) used as a cross-check, because
    # CodeCarbon on Windows without Intel Power Gadget assumes a constant 50% of a
    # generic 85 W TDP regardless of load.
    "cpu_power_w": 45.0,
    "cpu_power_source": "Intel i5-13500H Processor Base Power 45 W (max turbo 95 W).",
    "ram_power_w_per_8gb": 3.0,
    # Uncertainty factors on the energy total. CPU power is estimated, not metered.
    "energy_factor_low": 0.5,
    "energy_factor_high": 2.0,
    "unmeasured_overhead_high": 1.25,
    "uncertainty_basis": (
        "CPU energy is estimated (no RAPL/Power Gadget counter on this Windows laptop), so "
        "the total is bracketed at 0.5x-2x, then x1.25 on the high side for unmetered "
        "motherboard, storage, fans, display and charger losses. Emission bounds combine "
        "this with the carbon-intensity range. Engineering assumptions, not statistical "
        "confidence intervals."),
    "measurement_exclusions": (
        "Monitor, network equipment, lighting, embodied/manufacturing carbon, dataset "
        "download and report writing. Charger/battery conversion losses only in upper bound."),
}
