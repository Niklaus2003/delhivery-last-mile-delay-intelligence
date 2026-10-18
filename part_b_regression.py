
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.shipment_prep import FEATURES


def calculate_metrics(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, predictions),
    }


def main():
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BASE_DIR / "data" / "modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Split: 70% training, 15% validation, 15% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 2. Fit the diagnostic scaler on training rows only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    checks = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_scaled_max_abs_mean": np.abs(
            X_train_scaled.mean(axis=0)
        ).max(),
        "test_scaled_max_abs_mean": np.abs(
            X_test_scaled.mean(axis=0)
        ).max(),
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in checks.items()]
    ).to_csv(outputs_dir / "part_b_checks.csv", index=False, float_format="%.10f")

    # 3. Mean baseline.
    baseline_prediction = np.full(len(y_val), y_train.mean())
    validation_rows = [{
        "model": "mean_baseline",
        "split": "val",
        **calculate_metrics(y_val, baseline_prediction),
    }]

    # 4. Decision-tree depth diagnostic.
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_results = []
    tree_models = {}
    depth_order = {
        1: 1, 2: 2, 3: 3, 4: 4,
        6: 5, 8: 6, 12: 7, "unlimited": 8,
    }

    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(
            y_train, tree.predict(X_train)
        ))
        val_rmse = np.sqrt(mean_squared_error(
            y_val, tree.predict(X_val)
        ))

        depth_name = "unlimited" if depth is None else depth
        depth_results.append({
            "max_depth": depth_name,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })
        tree_models[depth_name] = tree

    depth_df = pd.DataFrame(depth_results)
    depth_df.to_csv(
        outputs_dir / "part_b_tree_depth.csv",
        index=False,
        float_format="%.4f",
    )

    best_tree_row = min(
        depth_results,
        key=lambda row: (
            row["val_rmse"],
            depth_order[row["max_depth"]],
        ),
    )
    best_tree = tree_models[best_tree_row["max_depth"]]

    # 5. Four candidate models. Fit only on training rows.
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "decision_tree": best_tree,
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_rows.append({
            "model": name,
            "split": "val",
            **calculate_metrics(y_val, model.predict(X_val)),
        })

    metrics_df = pd.DataFrame(validation_rows)

    # 6. Gradient Boosting learning-rate experiment.
    learning_rate_rows = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(
            n_estimators=100, learning_rate=lr, random_state=42
        )
        gb.fit(X_train, y_train)
        predictions = gb.predict(X_val)
        learning_rate_rows.append({
            "learning_rate": lr,
            "val_rmse": np.sqrt(mean_squared_error(y_val, predictions)),
            "val_r2": r2_score(y_val, predictions),
        })

    learning_rate_df = pd.DataFrame(learning_rate_rows)
    learning_rate_df.to_csv(
        outputs_dir / "part_b_gb_learning_rate.csv",
        index=False,
        float_format="%.4f",
    )

    # 7. Cross-validation for the two best validation candidates.
    ranked = metrics_df[
        metrics_df["model"].isin(models.keys())
    ].sort_values("RMSE")
    top_two = ranked["model"].head(2).tolist()

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    shared_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name in top_two:
        scores = cross_val_score(
            models[name],
            X_train_val,
            y_train_val,
            cv=shared_kfold,
        )
        cv_rows.append({
            "model": name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(
        outputs_dir / "part_b_cv.csv",
        index=False,
        float_format="%.4f",
    )

    # Select the final model by validation RMSE.
    best_validation = ranked.iloc[0]
    final_model_name = best_validation["model"]
    final_model = models[final_model_name]

    # Evaluate the selected model on test rows once.
    test_row = {
        "model": final_model_name,
        "split": "test",
        **calculate_metrics(y_test, final_model.predict(X_test)),
    }
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([test_row])],
        ignore_index=True,
    )
    metrics_df.to_csv(
        outputs_dir / "part_b_metrics.csv",
        index=False,
        float_format="%.4f",
    )

    # 8. Print summary diagnostic results
    depth_df["gap"] = depth_df["val_rmse"] - depth_df["train_rmse"]
    shallow = depth_df.loc[depth_df["max_depth"] == 1].iloc[0]
    unlimited = depth_df.loc[depth_df["max_depth"] == "unlimited"].iloc[0]
    best_depth = depth_df.loc[depth_df["val_rmse"].idxmin()]
    best_lr = learning_rate_df.loc[learning_rate_df["val_r2"].idxmax()]
    cv_winner = cv_df.loc[cv_df["mean_r2"].idxmax()]
    test_metrics = test_row

    print("Part B complete.")
    print("Split sizes:", len(X_train), len(X_val), len(X_test))
    print("Validation winner:", final_model_name)
    print("Validation RMSE:", round(best_validation["RMSE"], 4))
    print("Best tree depth:", best_depth["max_depth"], "with val RMSE:", round(best_depth["val_rmse"], 4))
    print("Best GB learning rate:", best_lr["learning_rate"], "with val R2:", round(best_lr["val_r2"], 4))
    print("CV favoured:", cv_winner["model"], "mean R2:", round(cv_winner["mean_r2"], 4))
    print("Test R2:", round(test_metrics["R2"], 4))
    print("Saved Part B outputs.")



if __name__ == "__main__":
    main()
