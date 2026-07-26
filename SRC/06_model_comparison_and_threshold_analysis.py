

from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_late_delivery_model.csv"
OUTPUT_DIR = ROOT / "data" / "output"
REPORT_DIR = ROOT / "reports"

MODEL_OUTPUT = OUTPUT_DIR / "model_comparison.csv"
THRESHOLD_OUTPUT = OUTPUT_DIR / "threshold_analysis.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "final_late_delivery_predictions.csv"
REPORT_OUTPUT = REPORT_DIR / "06_model_comparison_report.txt"

THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def evaluate(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "predicted_late_orders": int(prediction.sum()),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "order_id",
        "order_date",
        "late_delivery_risk",
        "days_for_shipment",
        "shipping_mode",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonne mancanti: {', '.join(missing)}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if df["order_date"].isna().any():
        raise ValueError("Sono presenti date ordine non valide.")

    orders = (
        df.groupby("order_id", as_index=False)
        .agg(
            order_date=("order_date", "first"),
            late_delivery_risk=("late_delivery_risk", "first"),
            days_for_shipment=("days_for_shipment", "first"),
            shipping_mode=("shipping_mode", "first"),
            order_region=("order_region", "first"),
            market=("market", "first"),
            customer_segment=("customer_segment", "first"),
            order_type=("type", "first"),
            order_value=("order_item_total", "sum"),
            order_profit=("order_profit_per_order", "sum"),
            quantity=("order_item_quantity", "sum"),
            discount=("order_item_discount", "sum"),
            unique_products=("order_item_cardprod_id", "nunique"),
            unique_categories=("category_id", "nunique"),
        )
        .sort_values("order_date")
        .reset_index(drop=True)
    )

    orders["order_month"] = orders["order_date"].dt.month
    orders["order_day_of_week"] = orders["order_date"].dt.dayofweek

    features = [
        "days_for_shipment",
        "shipping_mode",
        "order_region",
        "market",
        "customer_segment",
        "order_type",
        "order_value",
        "quantity",
        "discount",
        "unique_products",
        "unique_categories",
        "order_month",
        "order_day_of_week",
    ]

    split = int(len(orders) * 0.8)
    train, test = orders.iloc[:split], orders.iloc[split:]

    X_train, y_train = train[features], train["late_delivery_risk"]
    X_test, y_test = test[features], test["late_delivery_risk"]

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features if column not in numeric]

    linear_preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    tree_preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    models = {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            [
                ("preprocessing", linear_preprocessing),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessing", tree_preprocessing),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=15,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocessing", tree_preprocessing),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    probabilities = {}
    comparison_rows = []
    threshold_rows = []

    for name, model in models.items():
        start = perf_counter()
        model.fit(X_train, y_train)
        seconds = perf_counter() - start

        probability = model.predict_proba(X_test)[:, 1]
        probabilities[name] = probability

        metrics = evaluate(y_test, probability, 0.50)
        comparison_rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, probability),
                "fit_seconds": seconds,
                **metrics,
            }
        )

        if name != "Dummy Baseline":
            for threshold in THRESHOLDS:
                threshold_rows.append(
                    {
                        "model": name,
                        **evaluate(y_test, probability, threshold),
                    }
                )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["f1", "roc_auc"],
        ascending=False,
    )
    thresholds = pd.DataFrame(threshold_rows)

    best = thresholds.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]

    best_model = best["model"]
    best_threshold = float(best["threshold"])
    best_probability = probabilities[best_model]
    best_prediction = (best_probability >= best_threshold).astype(int)

    predictions = test[
        [
            "order_id",
            "order_date",
            "late_delivery_risk",
            "shipping_mode",
            "order_region",
            "market",
            "customer_segment",
            "order_value",
            "order_profit",
        ]
    ].copy()

    predictions["late_delivery_probability"] = best_probability
    predictions["predicted_late_delivery"] = best_prediction
    predictions["risk_level"] = pd.cut(
        best_probability,
        bins=[0, 0.40, 0.70, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    predictions["prediction_correct"] = (
        predictions["late_delivery_risk"] == predictions["predicted_late_delivery"]
    ).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(MODEL_OUTPUT, index=False, encoding="utf-8-sig")
    thresholds.to_csv(THRESHOLD_OUTPUT, index=False, encoding="utf-8-sig")
    predictions.to_csv(PREDICTIONS_OUTPUT, index=False, encoding="utf-8-sig")

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "06 - MODEL COMPARISON AND THRESHOLD ANALYSIS",
        "",
        f"Orders used: {len(orders):,}",
        f"Training orders: {len(train):,}",
        f"Test orders: {len(test):,}",
        f"Training period: {train['order_date'].min().date()} to {train['order_date'].max().date()}",
        f"Test period: {test['order_date'].min().date()} to {test['order_date'].max().date()}",
        "",
        "Model comparison at threshold 0.50:",
        comparison[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "fit_seconds",
            ]
        ].to_string(index=False),
        "",
        "Threshold analysis:",
        thresholds[
            [
                "model",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "predicted_late_orders",
            ]
        ].to_string(index=False),
        "",
        f"Selected model: {best_model}",
        f"Selected threshold: {best_threshold:.2f}",
        f"Selected precision: {best['precision']:.4f}",
        f"Selected recall: {best['recall']:.4f}",
        f"Selected F1: {best['f1']:.4f}",
        "",
        "Selection rule:",
        "Highest F1 score; recall and precision used as tie-breakers.",
        "",
        "Leakage controls:",
        "- One row per order",
        "- Time-based train/test split",
        "- Excluded delivery_status",
        "- Excluded real shipping days",
        "- Excluded shipping date",
        "- Excluded order_status",
        "- Excluded direct identifiers from model features",
        "",
        f"Model comparison: {MODEL_OUTPUT}",
        f"Threshold analysis: {THRESHOLD_OUTPUT}",
        f"Final predictions: {PREDICTIONS_OUTPUT}",
    ]

    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {MODEL_OUTPUT}")
    print(f"Creato: {THRESHOLD_OUTPUT}")
    print(f"Creato: {PREDICTIONS_OUTPUT}")
    print(f"Creato: {REPORT_OUTPUT}")
    print(f"Modello selezionato: {best_model} | Soglia: {best_threshold:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
