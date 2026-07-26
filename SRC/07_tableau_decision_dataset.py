

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ORDERS_INPUT = ROOT / "data" / "processed" / "dataco_orders.csv"
PREDICTIONS_INPUT = ROOT / "data" / "output" / "final_late_delivery_predictions.csv"
OUTPUT = ROOT / "data" / "output" / "tableau_operations_decision_data.csv"
REPORT = ROOT / "reports" / "07_tableau_dataset_report.txt"


def main() -> None:
    for path in (ORDERS_INPUT, PREDICTIONS_INPUT):
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    orders = pd.read_csv(ORDERS_INPUT, low_memory=False)
    predictions = pd.read_csv(PREDICTIONS_INPUT, low_memory=False)

    required_orders = {
        "order_id",
        "revenue",
        "order_profit_per_order",
        "delivery_timing_group",
        "shipping_mode",
        "order_region",
        "market",
        "customer_segment",
        "shipping_delay_days",
        "loss_flag",
    }
    required_predictions = {
        "order_id",
        "late_delivery_risk",
        "late_delivery_probability",
    }

    missing_orders = sorted(required_orders - set(orders.columns))
    missing_predictions = sorted(required_predictions - set(predictions.columns))

    if missing_orders:
        raise ValueError(f"Colonne mancanti in dataco_orders.csv: {', '.join(missing_orders)}")
    if missing_predictions:
        raise ValueError(
            "Colonne mancanti in final_late_delivery_predictions.csv: "
            + ", ".join(missing_predictions)
        )

    columns = [
        "order_id",
        "order_date",
        "shipping_date",
        "revenue",
        "order_profit_per_order",
        "delivery_timing_group",
        "shipping_mode",
        "order_region",
        "market",
        "customer_segment",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "shipping_delay_days",
        "quantity",
        "unique_products",
        "unique_categories",
        "order_lines",
        "loss_flag",
    ]
    columns = [column for column in columns if column in orders.columns]

    tableau = predictions[
        ["order_id", "late_delivery_risk", "late_delivery_probability"]
    ].merge(
        orders[columns],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    if tableau["revenue"].isna().any():
        raise ValueError("Alcuni ordini previsti non sono presenti nel dataset ordini.")

    tableau["prediction_030"] = (
        tableau["late_delivery_probability"] >= 0.30
    ).astype(int)
    tableau["prediction_040"] = (
        tableau["late_delivery_probability"] >= 0.40
    ).astype(int)

    tableau["risk_level"] = pd.cut(
        tableau["late_delivery_probability"],
        bins=[0, 0.30, 0.40, 0.70, 1],
        labels=["Low", "Watch", "High", "Critical"],
        include_lowest=True,
    )

    tableau["prediction_result_030"] = (
        tableau["late_delivery_risk"].astype(str)
        + "-"
        + tableau["prediction_030"].astype(str)
    ).map(
        {
            "1-1": "True Positive",
            "0-0": "True Negative",
            "0-1": "False Positive",
            "1-0": "False Negative",
        }
    )

    tableau["prediction_result_040"] = (
        tableau["late_delivery_risk"].astype(str)
        + "-"
        + tableau["prediction_040"].astype(str)
    ).map(
        {
            "1-1": "True Positive",
            "0-0": "True Negative",
            "0-1": "False Positive",
            "1-0": "False Negative",
        }
    )

    tableau["revenue_at_risk_030"] = tableau["revenue"].where(
        tableau["prediction_030"].eq(1), 0
    )
    tableau["revenue_at_risk_040"] = tableau["revenue"].where(
        tableau["prediction_040"].eq(1), 0
    )
    tableau["profit_at_risk_030"] = tableau["order_profit_per_order"].where(
        tableau["prediction_030"].eq(1), 0
    )
    tableau["profit_at_risk_040"] = tableau["order_profit_per_order"].where(
        tableau["prediction_040"].eq(1), 0
    )

    tableau["recommended_action"] = tableau["risk_level"].map(
        {
            "Low": "Standard Monitoring",
            "Watch": "Review if Capacity Allows",
            "High": "Priority Review",
            "Critical": "Immediate Intervention",
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    tableau.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "07 - TABLEAU DECISION DATASET",
        "",
        f"Rows exported: {len(tableau):,}",
        f"Columns exported: {tableau.shape[1]}",
        f"Date range: {tableau['order_date'].min()} to {tableau['order_date'].max()}",
        "",
        "Risk level distribution:",
        tableau["risk_level"].value_counts(dropna=False).to_string(),
        "",
        "Prediction results at threshold 0.30:",
        tableau["prediction_result_030"].value_counts().to_string(),
        "",
        "Prediction results at threshold 0.40:",
        tableau["prediction_result_040"].value_counts().to_string(),
        "",
        f"Output: {OUTPUT}",
    ]

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {OUTPUT}")
    print(f"Creato: {REPORT}")
    print(f"Righe: {len(tableau):,} | Colonne: {tableau.shape[1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
