"""Part D: Hub Segments and Separation Analysis for Delhivery Capstone.

Aggregates shipment performance into hub-level profiles, clusters the 24 hubs into 3 operating segments
using KMeans on standardized metrics, calculates separation ratios across K=2, 3, 4, 5, and identifies
tier-vs-segment mismatches.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the modelling table and hub metadata.
    df = pd.read_csv(DATA_DIR / "modelling_table.csv")
    hubs = pd.read_csv(DATA_DIR / "hubs.csv")

    # 2. Build the hub-level profile (sorted ascending by hub_id).
    hub_profile = (
        df.groupby("hub_id", sort=True)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    hub_profile_to_save = hub_profile.copy()
    hub_profile_to_save[
        ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    ] = hub_profile_to_save[
        ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    ].round(4)
    hub_profile_to_save.to_csv(OUTPUT_DIR / "hub_profile.csv", index=False)

    # 3. Scale the four profile columns in the required hub order.
    profile_features = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]

    X_profile = hub_profile[profile_features].to_numpy()
    X_scaled = StandardScaler().fit_transform(X_profile)

    # 4. Calculate separation evidence for K = 2, 3, 4, 5.
    distance_matrix = pairwise_distances(X_scaled, metric="euclidean")
    upper_i, upper_j = np.triu_indices(len(hub_profile), k=1)

    separation_rows = []

    for k in [2, 3, 4, 5]:
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = km.fit_predict(X_scaled)

        same_cluster = labels[upper_i] == labels[upper_j]
        different_cluster = ~same_cluster

        within_mean = distance_matrix[upper_i[same_cluster], upper_j[same_cluster]].mean()
        across_mean = distance_matrix[upper_i[different_cluster], upper_j[different_cluster]].mean()

        cluster_sizes = sorted(
            np.bincount(labels).tolist(),
            reverse=True,
        )

        separation_rows.append({
            "K": k,
            "across_over_within": round(float(across_mean / within_mean), 4),
            "cluster_sizes": "-".join(map(str, cluster_sizes)),
        })

    separation = pd.DataFrame(separation_rows)
    separation.to_csv(OUTPUT_DIR / "part_d_separation.csv", index=False)

    # 5. Fit the required three-cluster solution.
    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )
    cluster_labels = km.fit_predict(X_scaled)

    labelled_profile = hub_profile.copy()
    labelled_profile["cluster_id"] = cluster_labels

    # Name clusters by ascending mean hub delay rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay_rates = (
        labelled_profile.groupby("cluster_id")["delay_rate"].mean().sort_values()
    )
    segment_names = {
        cluster_delay_rates.index[0]: "Standard",
        cluster_delay_rates.index[1]: "Watch",
        cluster_delay_rates.index[2]: "Escalate",
    }

    labelled_profile["segment"] = labelled_profile["cluster_id"].map(segment_names)

    # 6. Save hub-to-segment assignments and hub metadata.
    hub_segments = labelled_profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one",
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id").reset_index(drop=True)

    hub_segments.to_csv(OUTPUT_DIR / "hub_segments.csv", index=False)

    # 7. Save segment-level averages of hub-level profiles.
    segment_order = ["Standard", "Watch", "Escalate"]

    segment_profile = (
        labelled_profile.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reindex(segment_order)
        .reset_index()
    )

    segment_profile[
        ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    ] = segment_profile[
        ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    ].round(4)
    segment_profile.to_csv(OUTPUT_DIR / "segment_profile.csv", index=False)

    # 8. Find hubs whose tier label disagrees with their segment: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    expected_segment = {
        1: "Standard",
        2: "Watch",
        3: "Escalate",
    }

    tier_mismatch = hub_segments[
        hub_segments.apply(
            lambda row: row["segment"] != expected_segment.get(row["hub_tier"]),
            axis=1,
        )
    ].reset_index(drop=True)

    tier_mismatch.to_csv(
        OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
        index=False,
    )

    # 9. Print checks and results.
    print("Hub profile shape:", hub_profile.shape)
    print("Shipments per hub:", sorted(hub_profile["shipments"].unique()))
    print("\nHub profile spot checks:")
    print(
        hub_profile[hub_profile["hub_id"].isin(["HUB01", "HUB13", "HUB17"])]
        .round(4)
        .to_string(index=False)
    )

    print("\nK-Means separation:")
    print(separation.to_string(index=False))

    print("\nSegment profiles:")
    print(segment_profile.to_string(index=False))

    print("\nTier-versus-segment mismatches:")
    print(tier_mismatch.to_string(index=False))

    print("\nSaved five Part D segmentation outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
