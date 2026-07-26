

from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_late_delivery_model.csv"
OUTPUT = ROOT / "data" / "output" / "late_delivery_predictions.csv"
REPORT = ROOT / "reports" / "05_late_delivery_model_report.txt"


def main() -> None:
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

    feature_columns = [
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
    train = orders.iloc[:split]
    test = orders.iloc[split:]

    X_train = train[feature_columns]
    y_train = train["late_delivery_risk"]
    X_test = test[feature_columns]
    y_test = test["late_delivery_risk"]

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in feature_columns if column not in numeric]

    preprocessing = ColumnTransformer(
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
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_prediction = baseline.predict(X_test)

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    results = test[
        ["order_id", "order_date", "late_delivery_risk"]
    ].copy()
    results["late_delivery_probability"] = probability
    results["predicted_late_delivery"] = prediction
    results["risk_level"] = pd.cut(
        probability,
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "05 - LATE DELIVERY PREDICTION",
        "",
        f"Orders used: {len(orders):,}",
        f"Training orders: {len(train):,}",
        f"Test orders: {len(test):,}",
        f"Training period: {train['order_date'].min().date()} to {train['order_date'].max().date()}",
        f"Test period: {test['order_date'].min().date()} to {test['order_date'].max().date()}",
        "",
        f"Baseline accuracy: {accuracy_score(y_test, baseline_prediction):.4f}",
        f"Logistic regression accuracy: {accuracy_score(y_test, prediction):.4f}",
        f"Logistic regression ROC-AUC: {roc_auc_score(y_test, probability):.4f}",
        "",
        "Classification report:",
        classification_report(y_test, prediction, digits=4),
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
        f"Predictions output: {OUTPUT}",
    ]

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {OUTPUT}")
    print(f"Creato: {REPORT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
