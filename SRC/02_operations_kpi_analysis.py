

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_operations_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "output"
REPORTS_DIR = ROOT / "reports"

KPI_FILE = OUTPUT_DIR / "operations_kpis.csv"
DETAIL_FILE = OUTPUT_DIR / "operations_detail.csv"
REPORT_FILE = REPORTS_DIR / "02_operations_analysis_report.txt"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "order_id",
        "revenue_value",
        "order_profit_per_order",
        "late_delivery_risk",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "shipping_delay_days",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonne mancanti: {', '.join(missing)}")

    orders = df["order_id"].nunique()
    revenue = df["revenue_value"].sum()
    profit = df["order_profit_per_order"].sum()
    late_rate = df["late_delivery_risk"].mean()
    on_time_rate = 1 - late_rate
    avg_real_days = df["days_for_shipping_real"].mean()
    avg_scheduled_days = df["days_for_shipment_scheduled"].mean()
    avg_delay = df["shipping_delay_days"].mean()
    loss_orders = df.loc[df["order_profit_per_order"] < 0, "order_id"].nunique()
    late_profit = df.loc[df["late_delivery_risk"] == 1, "order_profit_per_order"].sum()

    kpis = pd.DataFrame(
        {
            "kpi": [
                "Unique Orders",
                "Order Lines",
                "Revenue",
                "Profit",
                "Profit Margin",
                "Late Delivery Rate",
                "On-Time Delivery Rate",
                "Average Shipping Days",
                "Average Scheduled Shipping Days",
                "Average Delay Days",
                "Loss-Making Orders",
                "Profit from Late Deliveries",
            ],
            "value": [
                orders,
                len(df),
                revenue,
                profit,
                profit / revenue if revenue else 0,
                late_rate,
                on_time_rate,
                avg_real_days,
                avg_scheduled_days,
                avg_delay,
                loss_orders,
                late_profit,
            ],
        }
    )
    kpis.to_csv(KPI_FILE, index=False, encoding="utf-8-sig")

    dimensions = [
        column
        for column in [
            "shipping_mode",
            "market",
            "order_region",
            "category_name",
            "customer_segment",
        ]
        if column in df.columns
    ]

    summaries = []
    for dimension in dimensions:
        summary = (
            df.groupby(dimension, dropna=False)
            .agg(
                order_lines=("order_id", "size"),
                unique_orders=("order_id", "nunique"),
                revenue=("revenue_value", "sum"),
                profit=("order_profit_per_order", "sum"),
                late_delivery_rate=("late_delivery_risk", "mean"),
                avg_shipping_days=("days_for_shipping_real", "mean"),
                avg_delay_days=("shipping_delay_days", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "dimension", dimension)
        summary = summary.rename(columns={dimension: "dimension_value"})
        summary["profit_margin"] = summary["profit"].div(summary["revenue"]).fillna(0)
        summaries.append(summary)

    detail = pd.concat(summaries, ignore_index=True)
    detail.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    worst_late = (
        detail.loc[detail["unique_orders"] >= 100]
        .sort_values("late_delivery_rate", ascending=False)
        .head(10)
    )

    worst_profit = (
        detail.loc[detail["unique_orders"] >= 100]
        .sort_values("profit")
        .head(10)
    )

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "02 - OPERATIONS KPI ANALYSIS",
        "",
        f"Unique orders: {orders:,}",
        f"Order lines: {len(df):,}",
        f"Revenue: ${revenue:,.2f}",
        f"Profit: ${profit:,.2f}",
        f"Profit margin: {profit / revenue:.2%}",
        f"Late delivery rate: {late_rate:.2%}",
        f"On-time delivery rate: {on_time_rate:.2%}",
        f"Average shipping days: {avg_real_days:.2f}",
        f"Average scheduled shipping days: {avg_scheduled_days:.2f}",
        f"Average delay days: {avg_delay:.2f}",
        f"Loss-making orders: {loss_orders:,}",
        "",
        "Highest late-delivery groups (minimum 100 orders):",
        worst_late[
            ["dimension", "dimension_value", "unique_orders", "late_delivery_rate"]
        ].to_string(index=False),
        "",
        "Lowest-profit groups (minimum 100 orders):",
        worst_profit[
            ["dimension", "dimension_value", "unique_orders", "profit", "profit_margin"]
        ].to_string(index=False),
    ]
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {KPI_FILE}")
    print(f"Creato: {DETAIL_FILE}")
    print(f"Creato: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
