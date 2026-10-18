
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.shipment_prep import FEATURES, clean_shipments


def main():
    data_dir = BASE_DIR / "data"
    output_dir = BASE_DIR / "outputs"
    models_dir = BASE_DIR / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read the final configuration selected in Part C.
    config = pd.read_csv(
        output_dir / "part_c_final_config.csv"
    ).iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])

    max_depth_value = config["max_depth"]
    max_depth = (
        None
        if str(max_depth_value).strip().lower() == "unlimited"
        else int(max_depth_value)
    )

    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # 2. Load the modelling table and recreate the exact Part C split.
    df = pd.read_csv(data_dir / "modelling_table.csv")

    assert df["shipment_id"].is_unique, (
        "Shipment IDs in modelling_table.csv are not unique."
    )

    X = df[FEATURES]
    y = df["is_delayed"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    # 3. Build and fit the final model on training rows only.
    model = Pipeline([
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=selector_k,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    # 4. Save the trained model.
    model_path = models_dir / "delay_model.pkl"
    joblib.dump(model, model_path)

    # 5. Save the classification threshold.
    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv(
        models_dir / "delay_threshold.csv",
        index=False,
    )

    # 6. Save the model's required input feature order.
    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        models_dir / "feature_order.csv",
        index=False,
    )

    # 7. Load raw shipment records for imputation medians.
    raw_df = pd.read_csv(
        data_dir / "raw_shipments.csv"
    )

    # Remove only exact duplicate rows in memory.
    # The original raw_shipments.csv is not modified.
    raw_df = raw_df.drop_duplicates().copy()

    assert raw_df["shipment_id"].is_unique, (
        "Shipment IDs remain duplicated after exact duplicate removal."
    )

    # Calculate imputation medians on the full historical dataset (Part A rule)
    _, medians = clean_shipments(raw_df)

    expected_median_columns = {
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    }

    assert set(medians.keys()) == expected_median_columns, (
        f"Unexpected median columns: {list(medians.keys())}"
    )

    pd.DataFrame([
        {
            "column": column,
            "median": median,
        }
        for column, median in medians.items()
    ]).to_csv(
        models_dir / "impute_values.csv",
        index=False,
    )

    # 8. Reload the saved model and compare test probabilities.
    reloaded_model = joblib.load(model_path)

    in_memory_probs = model.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_model.predict_proba(X_test)[:, 1]

    max_abs_diff = float(
        np.max(np.abs(in_memory_probs - reloaded_probs))
    )

    reload_check = pd.DataFrame([
        {
            "metric": "test_rows_scored",
            "value": len(X_test),
        },
        {
            "metric": "reload_max_abs_probability_diff",
            "value": max_abs_diff,
        },
        {
            "metric": "threshold",
            "value": threshold,
        },
    ])

    reload_check.to_csv(
        output_dir / "part_d_reload_check.csv",
        index=False,
    )

    # 9. Display results.
    print("Model saved:", model_path)
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows scored:", len(X_test))
    print("Features in original order:", len(FEATURES))
    print("Selector k:", selector_k)
    print("Threshold:", threshold)

    print("\nRaw data checks:")
    print("Raw rows after exact duplicate removal:", len(raw_df))
    print("Training raw rows used for medians:", len(raw_train))

    print("\nImputation medians:")
    print(
        pd.DataFrame(
            list(medians.items()),
            columns=["column", "median"],
        ).to_string(index=False)
    )

    print("\nReload check:")
    print(reload_check.to_string(index=False))

    # 10. Validate the packaging results.
    assert len(X_train) == 840, (
        "Unexpected training-set size."
    )
    assert len(X_val) == 180, (
        "Unexpected validation-set size."
    )
    assert len(X_test) == 180, (
        "Unexpected test-set size."
    )
    assert max_abs_diff <= 1e-9, (
        "Reloaded model probabilities differ."
    )

    saved_features = pd.read_csv(
        models_dir / "feature_order.csv"
    )["feature"].tolist()

    assert saved_features == FEATURES, (
        "Saved feature order does not match FEATURES."
    )

    saved_medians = pd.read_csv(
        models_dir / "impute_values.csv"
    )

    assert set(saved_medians["column"]) == expected_median_columns, (
        "Saved imputation median columns are incorrect."
    )

    assert saved_medians["median"].notna().all(), (
        "One or more saved medians are missing."
    )

    print("\nPASS: Model packaging and reload checks succeeded.")


if __name__ == "__main__":
    main()
