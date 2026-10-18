import numpy as np
import pandas as pd



PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod",
    "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):
  clean_df = raw.drop_duplicates().copy()

  clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()

  clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

  clean_df["declared_value_inr"] = pd.to_numeric(
    clean_df["declared_value"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
  )

  clean_df = clean_df.drop(columns=["declared_value"])

  clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"], errors="coerce")

  clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

  median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

  if medians is None:
        medians = {
            column: float(clean_df[column].median())
            for column in median_columns
        }
  else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

  for column in median_columns:
        clean_df[column] = clean_df[column].fillna(medians[column])

  return clean_df, medians


def build_features(clean, hubs):
    result = clean.merge(
        hubs[["hub_id", "hub_city", "region",
              "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left"
    )

    result["promised_hours"] = (
        result["service_type"].map(PROMISED_HOURS).astype(int)
    )

    result["is_cod"] = (
        result["payment_mode"].eq("COD").astype(int)
    )

    result["is_weekend"] = (
        result["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    result["dispatch_hour"] = result["dispatch_time"].dt.hour.astype(int)

    result["log_declared_value"] = np.log(result["declared_value_inr"])

    return result
