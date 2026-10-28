from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.shipment_prep import FEATURES, clean_shipments, build_features


def main():
    data_dir = BASE_DIR / "data"
    output_dir = BASE_DIR / "outputs"
    models_dir = BASE_DIR / "models"

    output_dir.mkdir(parents=True, exist_ok=True)

    new_shipments = pd.read_csv(data_dir / "new_shipments.csv")
    hubs = pd.read_csv(data_dir / "hubs.csv")
    hub_segments = pd.read_csv(output_dir / "hub_segments.csv")

    model = joblib.load(models_dir / "delay_model.pkl")
    threshold_df = pd.read_csv(models_dir / "delay_threshold.csv")
    threshold = float(threshold_df.loc[0, "threshold"])

    feature_order = pd.read_csv(models_dir / "feature_order.csv")["feature"].tolist()
    impute_values = pd.read_csv(models_dir / "impute_values.csv")
    medians = dict(zip(impute_values["column"], impute_values["median"]))

    input_shipment_ids = new_shipments["shipment_id"].tolist()

    clean_df, _ = clean_shipments(new_shipments.copy(), medians=medians)
    feature_df = build_features(clean_df, hubs)

    X_new = feature_df[feature_order].copy()
    probabilities = model.predict_proba(X_new)[:, 1]
    flagged = (probabilities >= threshold).astype(int)

    segment_lookup = hub_segments[["hub_id", "segment"]].copy()

    results = pd.DataFrame({
        "shipment_id": input_shipment_ids,
        "hub_id": clean_df["hub_id"].tolist(),
        "delay_probability": np.round(probabilities, 4),
        "flagged": flagged,
    })

    results = results.merge(
        segment_lookup,
        on="hub_id",
        how="left",
        sort=False,
    )

    results["action"] = np.select(
        [
            (results["flagged"] == 1) & (results["segment"] == "Escalate"),
            results["flagged"] == 1,
        ],
        [
            "Escalate to hub manager",
            "Proactive customer call",
        ],
        default="No action",
    )

    results = results[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action",
        ]
    ]

    output_path = output_dir / "new_shipment_actions.csv"
    results.to_csv(output_path, index=False)

    print("Predictions completed successfully.")


if __name__ == "__main__":
    main()
