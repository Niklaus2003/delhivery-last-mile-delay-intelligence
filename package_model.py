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

    config = pd.read_csv(output_dir / "part_c_final_config.csv").iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_val = config["max_depth"]
    max_depth = None if str(max_depth_val).strip().lower() == "unlimited" else int(max_depth_val)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    df = pd.read_csv(data_dir / "modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    model_path = models_dir / "delay_model.pkl"
    joblib.dump(model, model_path)

    pd.DataFrame({"threshold": [threshold]}).to_csv(models_dir / "delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(models_dir / "feature_order.csv", index=False)

    raw_df = pd.read_csv(data_dir / "raw_shipments.csv")
    raw_df = raw_df.drop_duplicates().copy()
    _, medians = clean_shipments(raw_df)

    pd.DataFrame([
        {"column": col, "median": med} for col, med in medians.items()
    ]).to_csv(models_dir / "impute_values.csv", index=False)

    reloaded_model = joblib.load(model_path)
    in_memory_probs = model.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_model.predict_proba(X_test)[:, 1]
    max_abs_diff = float(np.max(np.abs(in_memory_probs - reloaded_probs)))

    reload_check = pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": max_abs_diff},
        {"metric": "threshold", "value": threshold},
    ])
    reload_check.to_csv(output_dir / "part_d_reload_check.csv", index=False)

    assert max_abs_diff <= 1e-9
    print("Packaging completed successfully.")


if __name__ == "__main__":
    main()
