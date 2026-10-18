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
        modelling_table["delivery_hours"]
        > modelling_table["promised_hours"]
    ).astype(int)

    deduplicated = raw.drop_duplicates().copy()

    service_type_spellings_before = deduplicated["service_type"].nunique()
    service_type_spellings_after = clean_df["service_type"].nunique()

    missing_distance_before = deduplicated["distance_km"].isna().sum()
    missing_weight_before = deduplicated["parcel_weight_kg"].isna().sum()
    sentinel_weight_rows = (
        deduplicated["parcel_weight_kg"] > 100
    ).sum()
    missing_declared_value_before = deduplicated["declared_value"].isna().sum()
    missing_payment_mode_before = deduplicated["payment_mode"].isna().sum()

    hours_q1 = modelling_table["delivery_hours"].quantile(0.25)
    hours_q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = hours_q3 - hours_q1
    hours_lower_bound = hours_q1 - 1.5 * iqr
    hours_upper_bound = hours_q3 + 1.5 * iqr
    hours_outlier_count = (
        (modelling_table["delivery_hours"] < hours_lower_bound)
        | (modelling_table["delivery_hours"] > hours_upper_bound)
    ).sum()

    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(deduplicated),
        "rows_after_dedup": len(deduplicated),
        "missing_distance_km_before": missing_distance_before,
        "missing_parcel_weight_before": missing_weight_before,
        "sentinel_weight_rows": sentinel_weight_rows,
        "missing_declared_value_before": missing_declared_value_before,
        "missing_payment_mode_before": missing_payment_mode_before,
        "service_type_spellings_before": service_type_spellings_before,
        "service_type_spellings_after": service_type_spellings_after,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(modelling_table),
        "nulls_in_features_and_target": modelling_table[
            FEATURES + ["delivery_hours", "is_delayed"]
        ].isna().sum().sum(),
        "delayed_count": modelling_table["is_delayed"].sum(),
        "delay_rate": modelling_table["is_delayed"].mean(),
        "hours_q1": hours_q1,
        "hours_q3": hours_q3,
        "hours_lower_bound": hours_lower_bound,
        "hours_upper_bound": hours_upper_bound,
        "hours_outlier_count": hours_outlier_count,
        "declared_value_mean_inr": modelling_table["declared_value_inr"].mean(),
        "declared_value_median_inr": modelling_table["declared_value_inr"].median(),
        "log_declared_value_mean": modelling_table["log_declared_value"].mean(),
        "log_declared_value_median": modelling_table["log_declared_value"].median(),
    }

    checks_df = pd.DataFrame([
        {
            "metric": metric,
            "value": round(value, 4)
            if isinstance(value, (float, np.floating))
            else value,
        }
        for metric, value in checks.items()
    ])
    checks_df.to_csv(BASE_DIR / "outputs" / "part_a_checks.csv", index=False)

    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True,
    )

    group_results = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        grouped = modelling_table.groupby(
            dimension, observed=False
        ).agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index()
        grouped = grouped.rename(columns={dimension: "group"})
        grouped["dimension"] = dimension
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

    modelling_table_to_save = modelling_table.copy()
    numeric_columns = modelling_table_to_save.select_dtypes(
        include=np.number
    ).columns
    modelling_table_to_save[numeric_columns] = (
        modelling_table_to_save[numeric_columns].round(4)
    )
    modelling_table_to_save.to_csv(
        BASE_DIR / "data" / "modelling_table.csv",
        index=False,
    )

    delivery_correlations = correlation_matrix["delivery_hours"].drop(
        "delivery_hours"
    )
    strongest_correlations = delivery_correlations.loc[
        delivery_correlations.abs().sort_values(ascending=False).index
    ].head(3)

    # Summary of findings printed to stdout
    service_results = delay_by_group[delay_by_group["dimension"] == "service_type"]
    tier_results = delay_by_group[delay_by_group["dimension"] == "hub_tier"]
    distance_results = delay_by_group[delay_by_group["dimension"] == "distance_band"]

    print("Part A complete.")
    print("Raw rows:", len(raw))
    print("Cleaned rows:", len(clean_df))
    print("Modelling rows:", len(modelling_table))
    print("Delayed shipments:", int(modelling_table["is_delayed"].sum()))
    print("Delay rate:", round(modelling_table["is_delayed"].mean(), 4))
    print("Medians:", medians)
    print("Strongest correlations with delivery_hours:")
    print(strongest_correlations.round(4).to_string())
    print("Saved Part A outputs.")


if __name__ == "__main__":
    main()
