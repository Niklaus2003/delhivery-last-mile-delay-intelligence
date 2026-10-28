from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.shipment_prep import (
    PROMISED_HOURS,
    FEATURES,
    clean_shipments,
    build_features,
)


def main():
    raw = pd.read_csv(BASE_DIR / "data" / "raw_shipments.csv")
    hubs = pd.read_csv(BASE_DIR / "data" / "hubs.csv")

    clean_df, medians = clean_shipments(raw)
    modelling_table = build_features(clean_df, hubs)

    modelling_table["is_delayed"] = (
        modelling_table["delivery_hours"] > modelling_table["promised_hours"]
    ).astype(int)

    dedup = raw.drop_duplicates().copy()

    service_before = dedup["service_type"].nunique()
    service_after = clean_df["service_type"].nunique()

    missing_dist = dedup["distance_km"].isna().sum()
    missing_weight = dedup["parcel_weight_kg"].isna().sum()
    sentinel_weight = (dedup["parcel_weight_kg"] > 100).sum()
    missing_val = dedup["declared_value"].isna().sum()
    missing_pay = dedup["payment_mode"].isna().sum()

    q1 = modelling_table["delivery_hours"].quantile(0.25)
    q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = (
        (modelling_table["delivery_hours"] < lower_bound)
        | (modelling_table["delivery_hours"] > upper_bound)
    ).sum()

    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(dedup),
        "rows_after_dedup": len(dedup),
        "missing_distance_km_before": missing_dist,
        "missing_parcel_weight_before": missing_weight,
        "sentinel_weight_rows": sentinel_weight,
        "missing_declared_value_before": missing_val,
        "missing_payment_mode_before": missing_pay,
        "service_type_spellings_before": service_before,
        "service_type_spellings_after": service_after,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(modelling_table),
        "nulls_in_features_and_target": modelling_table[
            FEATURES + ["delivery_hours", "is_delayed"]
        ].isna().sum().sum(),
        "delayed_count": modelling_table["is_delayed"].sum(),
        "delay_rate": modelling_table["is_delayed"].mean(),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower_bound,
        "hours_upper_bound": upper_bound,
        "hours_outlier_count": outlier_count,
        "declared_value_mean_inr": modelling_table["declared_value_inr"].mean(),
        "declared_value_median_inr": modelling_table["declared_value_inr"].median(),
        "log_declared_value_mean": modelling_table["log_declared_value"].mean(),
        "log_declared_value_median": modelling_table["log_declared_value"].median(),
    }

    checks_df = pd.DataFrame([
        {
            "metric": k,
            "value": round(v, 4) if isinstance(v, (float, np.floating)) else v,
        }
        for k, v in checks.items()
    ])
    (BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    checks_df.to_csv(BASE_DIR / "outputs" / "part_a_checks.csv", index=False)

    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True,
    )

    group_results = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        grouped = modelling_table.groupby(dim, observed=False).agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index()
        grouped = grouped.rename(columns={dim: "group"})
        grouped["dimension"] = dim
        group_results.append(grouped)

    delay_by_group = pd.concat(group_results, ignore_index=True)
    delay_by_group = delay_by_group[
        ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
    ]
    delay_by_group[["delay_rate", "mean_hours"]] = (
        delay_by_group[["delay_rate", "mean_hours"]].round(4)
    )
    delay_by_group.to_csv(
        BASE_DIR / "outputs" / "part_a_delay_by_group.csv",
        index=False,
    )

    numeric_table = modelling_table.select_dtypes(include=np.number)
    correlation_matrix = numeric_table.corr()
    correlation_matrix.round(4).to_csv(
        BASE_DIR / "outputs" / "part_a_correlations.csv",
        index=True,
        index_label="column",
    )

    save_table = modelling_table.copy()
    num_cols = save_table.select_dtypes(include=np.number).columns
    save_table[num_cols] = save_table[num_cols].round(4)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    save_table.to_csv(BASE_DIR / "data" / "modelling_table.csv", index=False)

    print("Part A completed successfully.")


if __name__ == "__main__":
    main()
