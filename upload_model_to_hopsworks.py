"""
Upload the saved winning model + feature config to Hopsworks Model Registry.

Assumes you already ran training and produced:
  out/model_artifacts/best_model_pipeline.joblib
  out/model_artifacts/best_model_config.json

Login credentials are picked up the same way as in your notebook:
  - set HOPSWORKS_API_KEY (and optionally HOPSWORKS_PROJECT) in your environment or .env
  - then hopsworks.login(engine="python")
"""

from pathlib import Path
import json
import hopsworks

# ----------------- SETTINGS -----------------
MODEL_DIR = Path("out/model_artifacts")
MODEL_FILE = MODEL_DIR / "best_model_pipeline.joblib"
CONFIG_FILE = MODEL_DIR / "best_model_config.json"

# Name/version in the Model Registry (change if you want)
MODEL_NAME = "occupancy_rf"
MODEL_VERSION = None  # set an int like 1 if you want to control versioning yourself

# Optional metadata
DESCRIPTION = "RandomForest pipeline for occupancy label_grouped prediction (saved with preprocessing + winning feature config)."
# -------------------------------------------


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_FILE}")
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config artifact: {CONFIG_FILE}")

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    # Login (same style as your notebook)
    project = hopsworks.login(engine="python")
    mr = project.get_model_registry()

    # Metrics: Hopsworks is happy with a dict of floats.
    # Your config contains metrics_on_test; we pass them through.
    metrics = config.get("metrics_on_test", {})
    # If you used balanced_acc as your "accuracy", keep it explicit.
    # If you computed plain accuracy separately, include it here too.
    # Example:
    # metrics["accuracy"] = config["metrics_on_test"].get("accuracy")

    # Register model metadata (Sklearn is appropriate since it's a sklearn Pipeline)
    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        metrics=metrics,
        description=DESCRIPTION,
    )

    # Upload the whole directory so BOTH files go into the model artifact:
    #  - best_model_pipeline.joblib
    #  - best_model_config.json
    model.save(str(MODEL_DIR))

    print("Uploaded to Hopsworks Model Registry:")
    print("  name:", MODEL_NAME)
    print("  version:", model.version)
    print("  artifacts:", str(MODEL_DIR))
    print("  metrics:", metrics)
    print("  winning_features:", config.get("winning_features"))


if __name__ == "__main__":
    main()
