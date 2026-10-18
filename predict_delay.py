
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

    # 1. Load new shipments, hub information, and model artifacts.
    new_shipments = pd.read_csv(data_dir / "new_shipments.csv")
    hubs = pd.read_csv(data_dir / "hubs.csv")
    hub_segments = pd.read_csv(output_dir / "hub_segments.csv")

    model = joblib.load(models_dir / "delay_model.pkl")

    threshold_df = pd.read_csv(
        models_dir / "delay_threshold.csv"
    )
    threshold = float(threshold_df.loc[0, "threshold"])

    feature_order = pd.read_csv(
        models_dir / "feature_order.csv"
    )["feature"].tolist()

    impute_values = pd.read_csv(
        models_dir / "impute_values.csv"
    )

    medians = dict(
        zip(impute_values["column"], impute_values["median"])
    )

    # 2. Validate required input columns and preserve input order.
    required_columns = [
        "shipment_id",
        "hub_id",
        "service_type",
        "payment_mode",
        "parcel_weight_kg",
        "distance_km",
        "declared_value",
        "n_stops",
        "rain_flag",
        "sale_event_flag",
        "dispatch_time",
        "label_print_batch",
        "scanner_build_no",
        "tape_roll_no",
        "sticker_serial_mod",
        "shelf_slot_no",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in new_shipments.columns
    ]

    assert not missing_columns, (
        f"Missing columns in new_shipments.csv: {missing_columns}"
    )

    assert new_shipments["shipment_id"].is_unique, (
        "New shipment IDs must be unique."
    )

    input_shipment_ids = new_shipments["shipment_id"].tolist()

    # 3. Clean the raw records using training-set medians.
    clean_df, _ = clean_shipments(
        new_shipments.copy(),
        medians=medians,
    )

    # Ensure cleaning has not removed or reordered input shipments.
    assert len(clean_df) == len(new_shipments), (
        "Cleaning changed the number of new shipments."
    )

    assert clean_df["shipment_id"].tolist() == input_shipment_ids, (
        "Cleaning changed shipment order."
    )

    # 4. Build the same features used during model training.
    feature_df = build_features(clean_df, hubs)

    # Check that the required model features exist.
    missing_features = [
        feature
        for feature in feature_order
        if feature not in feature_df.columns
    ]

    assert not missing_features, (
        f"Missing model features: {missing_features}"
    )

    X_new = feature_df[feature_order].copy()

    assert feature_order == FEATURES, (
        "Saved feature order differs from the training feature list."
    )

    assert X_new.notna().all().all(), (
        "Missing values remain in the model input features."
    )

    # 5. Predict delay probabilities and apply the saved threshold.
    probabilities = model.predict_proba(X_new)[:, 1]
    flagged = (probabilities >= threshold).astype(int)

    # 6. Attach the segment assigned to each hub.
    segment_lookup = hub_segments[
        ["hub_id", "segment"]
    ].copy()

    assert segment_lookup["hub_id"].is_unique, (
        "Each hub must have exactly one segment."
    )

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
        validate="many_to_one",
    )

    assert results["segment"].notna().all(), (
        "Some shipments have no matching hub segment."
    )

    # 7. Assign operational actions.
    results["action"] = np.select(
        [
            (results["flagged"] == 1)
            & (results["segment"] == "Escalate"),

            results["flagged"] == 1,
        ],
        [
            "Escalate to hub manager",
            "Proactive customer call",
        ],
        default="No action",
    )

    # 8. Keep the required output columns and original input order.
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

    assert len(results) == len(new_shipments), (
        "Prediction output does not contain one row per input shipment."
    )

    assert results["shipment_id"].tolist() == input_shipment_ids, (
        "Prediction output order differs from input order."
    )

    # 9. Save predictions.
    output_path = output_dir / "new_shipment_actions.csv"

    results.to_csv(
        output_path,
        index=False,
    )

    # 10. Print summary and sample results.
    print("Input shipments:", len(new_shipments))
    print("Output shipments:", len(results))
    print("Threshold:", threshold)
    print("Flagged shipments:", int(results["flagged"].sum()))
    print(
        "Escalations:",
        int((results["action"] == "Escalate to hub manager").sum()),
    )
    print(
        "Proactive customer calls:",
        int((results["action"] == "Proactive customer call").sum()),
    )
    print(
        "No-action shipments:",
        int((results["action"] == "No action").sum()),
    )

    print("\nAction counts:")
    print(results["action"].value_counts().to_string())

    print("\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))

    print("\nSaved:", output_path)
    print("\nPASS: New-shipment predictions completed successfully.")


if __name__ == "__main__":
    main()
