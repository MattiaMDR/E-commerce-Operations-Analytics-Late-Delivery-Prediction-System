

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_orders.csv"
OUTPUT_DIR = ROOT / "data" / "output"
REPORT_DIR = ROOT / "reports"

KPI_OUTPUT = OUTPUT_DIR / "order_level_kpis.csv"
BREAKDOWN_OUTPUT = OUTPUT_DIR / "order_level_breakdown.csv"
REPORT_OUTPUT = REPORT_DIR / "04_order_level_kpi_report.txt"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "order_id",
        "revenue",
        "order_profit_per_order",
        "delivery_timing_group",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "shipping_delay_days",
        "loss_flag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonne mancanti: {', '.join(missing)}")

    revenue = df["revenue"].sum()
    profit = df["order_profit_per_order"].sum()
    status_rate = df["delivery_timing_group"].value_counts(normalize=True)

    kpis = {
        "Unique Orders": df["order_id"].nunique(),
        "Revenue": revenue,
        "Profit": profit,
        "Profit Margin": profit / revenue if revenue else 0,
        "Late Delivery Rate": status_rate.get("Late", 0),
        "On-Time Delivery Rate": status_rate.get("On Time", 0),
        "Early Delivery Rate": status_rate.get("Early", 0),
        "Cancellation Rate": status_rate.get("Canceled", 0),
        "Average Shipping Days": df["days_for_shipping_real"].mean(),
        "Average Scheduled Shipping Days": df["days_for_shipment_scheduled"].mean(),
        "Average Delay Days": df["shipping_delay_days"].mean(),
        "Loss-Making Orders": int(df["loss_flag"].sum()),
        "Loss-Making Order Rate": df["loss_flag"].mean(),
        "Late Orders Revenue": df.loc[
            df["delivery_timing_group"].eq("Late"), "revenue"
        ].sum(),
        "Late Orders Profit": df.loc[
            df["delivery_timing_group"].eq("Late"), "order_profit_per_order"
        ].sum(),
    }

    pd.DataFrame(kpis.items(), columns=["kpi", "value"]).to_csv(
        KPI_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    dimensions = [
        column
        for column in (
            "shipping_mode",
            "market",
            "order_region",
            "customer_segment",
        )
        if column in df.columns
    ]

    breakdowns = []
    for dimension in dimensions:
        grouped = (
            df.groupby(dimension, dropna=False)
            .agg(
                unique_orders=("order_id", "nunique"),
                revenue=("revenue", "sum"),
                profit=("order_profit_per_order", "sum"),
                late_delivery_rate=(
                    "delivery_timing_group",
                    lambda values: values.eq("Late").mean(),
                ),
                on_time_delivery_rate=(
                    "delivery_timing_group",
                    lambda values: values.eq("On Time").mean(),
                ),
                cancellation_rate=(
                    "delivery_timing_group",
                    lambda values: values.eq("Canceled").mean(),
                ),
                loss_order_rate=("loss_flag", "mean"),
                avg_shipping_days=("days_for_shipping_real", "mean"),
                avg_delay_days=("shipping_delay_days", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "dimension_value"})
        )

        grouped.insert(0, "dimension", dimension)
        grouped["profit_margin"] = grouped["profit"].div(grouped["revenue"]).fillna(0)
        breakdowns.append(grouped)

    breakdown = pd.concat(breakdowns, ignore_index=True)
    breakdown.to_csv(
        BREAKDOWN_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    relevant = breakdown[breakdown["unique_orders"].ge(100)]

    worst_late = relevant.nlargest(10, "late_delivery_rate")
    worst_margin = relevant.nsmallest(10, "profit_margin")

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "04 - ORDER-LEVEL KPI ANALYSIS",
        "",
        f"Unique orders: {kpis['Unique Orders']:,}",
        f"Revenue: ${kpis['Revenue']:,.2f}",
        f"Profit: ${kpis['Profit']:,.2f}",
        f"Profit margin: {kpis['Profit Margin']:.2%}",
        f"Late delivery rate: {kpis['Late Delivery Rate']:.2%}",
        f"On-time delivery rate: {kpis['On-Time Delivery Rate']:.2%}",
        f"Early delivery rate: {kpis['Early Delivery Rate']:.2%}",
        f"Cancellation rate: {kpis['Cancellation Rate']:.2%}",
        f"Average shipping days: {kpis['Average Shipping Days']:.2f}",
        f"Average scheduled shipping days: {kpis['Average Scheduled Shipping Days']:.2f}",
        f"Average delay days: {kpis['Average Delay Days']:.2f}",
        f"Loss-making orders: {kpis['Loss-Making Orders']:,}",
        f"Loss-making order rate: {kpis['Loss-Making Order Rate']:.2%}",
        f"Late orders revenue: ${kpis['Late Orders Revenue']:,.2f}",
        f"Late orders profit: ${kpis['Late Orders Profit']:,.2f}",
        "",
        "Highest late-delivery groups (minimum 100 orders):",
        worst_late[
            ["dimension", "dimension_value", "unique_orders", "late_delivery_rate"]
        ].to_string(index=False),
        "",
        "Lowest-margin groups (minimum 100 orders):",
        worst_margin[
            ["dimension", "dimension_value", "unique_orders", "profit_margin"]
        ].to_string(index=False),
    ]

    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {KPI_OUTPUT}")
    print(f"Creato: {BREAKDOWN_OUTPUT}")
    print(f"Creato: {REPORT_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
