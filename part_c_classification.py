from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def compute_classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


def main():
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BASE_DIR / "data" / "modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    blind_preds = np.zeros(len(y_val), dtype=int)
    blind_metrics = compute_classification_metrics(y_val, blind_preds)
    blind_metrics["precision"] = 0.0000

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_val)
    lr_metrics = compute_classification_metrics(y_val, lr_preds)

    depths = [2, 3, 4, 5, 6, 8, 10]
    best_dt_depth = None
    best_dt_f1 = -1.0
    best_dt_metrics = None

    for d in depths:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_val)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_dt_f1:
            best_dt_f1 = f1
            best_dt_depth = d
            best_dt_metrics = compute_classification_metrics(y_val, preds)

    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    rf_default_preds = rf_default.predict(X_val)
    rf_default_metrics = compute_classification_metrics(y_val, rf_default_preds)

    models_table = pd.DataFrame([
        {"model": "blind_all_on_time", **blind_metrics},
        {"model": "logistic_regression", **lr_metrics},
        {"model": f"decision_tree_depth_{best_dt_depth}", **best_dt_metrics},
        {"model": "random_forest_default", **rf_default_metrics},
    ])
    models_table.to_csv(outputs_dir / "part_c_models.csv", index=False)

    rf_17 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_17.fit(X_train, y_train)
    rf_importances = rf_17.feature_importances_

    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": rf_importances,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    feature_scores_df["f_rank"] = (
        feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    )
    feature_scores_df = feature_scores_df.sort_values("f_rank").reset_index(drop=True)
    feature_scores_df = feature_scores_df[
        ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
    ]
    feature_scores_df["f_score"] = feature_scores_df["f_score"].round(4)
    feature_scores_df["rf_importance"] = feature_scores_df["rf_importance"].round(4)
    feature_scores_df.to_csv(outputs_dir / "part_c_feature_scores.csv", index=False)

    k_values = [5, 7, 10, 12, 17]
    selectk_rows = []
    for k in k_values:
        pipe_k = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipe_k.fit(X_train, y_train)
        val_preds_k = pipe_k.predict(X_val)
        val_f1_k = f1_score(y_val, val_preds_k, zero_division=0)
        selectk_rows.append({"k": k, "val_f1": round(float(val_f1_k), 4)})

    selectk_df = pd.DataFrame(selectk_rows)
    selectk_df.to_csv(outputs_dir / "part_c_selectk.csv", index=False)

    selector_7 = SelectKBest(score_func=f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
    )
    grid_search.fit(X_train_7, y_train)
    grid_val_preds = grid_search.predict(X_val_7)
    grid_val_f1 = f1_score(y_val, grid_val_preds, zero_division=0)

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    random_search.fit(X_train_7, y_train)
    rand_val_preds = random_search.predict(X_val_7)
    rand_val_f1 = f1_score(y_val, rand_val_preds, zero_division=0)

    default_forest_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    default_forest_7.fit(X_train_7, y_train)
    default_val_preds = default_forest_7.predict(X_val_7)
    default_val_f1 = f1_score(y_val, default_val_preds, zero_division=0)

    tuning_df = pd.DataFrame([
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(float(grid_search.best_score_), 4),
            "val_f1": round(float(grid_val_f1), 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(random_search.best_params_),
            "best_cv_accuracy": round(float(random_search.best_score_), 4),
            "val_f1": round(float(rand_val_f1), 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": round(float(default_val_f1), 4),
        },
    ])
    tuning_df.to_csv(outputs_dir / "part_c_tuning.csv", index=False)

    chosen_row = tuning_df.sort_values(by=["val_f1"], ascending=False).iloc[0]
    chosen_from = chosen_row["search"]

    if chosen_from == "default_forest_same_7_features":
        final_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    elif chosen_from == "grid_search":
        final_params = grid_search.best_params_
    else:
        final_params = random_search.best_params_

    final_model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **final_params)),
    ])
    final_model.fit(X_train, y_train)

    val_probs = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_rows = []

    for t in thresholds:
        preds_t = (val_probs >= t).astype(int)
        metrics_t = compute_classification_metrics(y_val, preds_t)
        threshold_rows.append({"threshold": round(t, 4), **metrics_t})

    thresholds_df = pd.DataFrame(threshold_rows)
    thresholds_df.to_csv(outputs_dir / "part_c_thresholds.csv", index=False)

    eligible_thresholds = thresholds_df[thresholds_df["recall"] >= 0.80]
    if not eligible_thresholds.empty:
        chosen_threshold = eligible_thresholds["threshold"].max()
    else:
        chosen_threshold = 0.1

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    test_metrics = compute_classification_metrics(y_test, test_preds)

    final_test_df = pd.DataFrame([{
        "threshold": round(chosen_threshold, 4),
        **test_metrics,
    }])
    final_test_df.to_csv(outputs_dir / "part_c_final_test.csv", index=False)

    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": final_params["n_estimators"],
        "max_depth": "unlimited" if final_params.get("max_depth") is None else final_params["max_depth"],
        "min_samples_leaf": final_params["min_samples_leaf"],
        "threshold": round(chosen_threshold, 4),
        "chosen_from": chosen_from,
    }])
    final_config_df.to_csv(outputs_dir / "part_c_final_config.csv", index=False)

    print("Part C completed successfully.")


if __name__ == "__main__":
    main()
