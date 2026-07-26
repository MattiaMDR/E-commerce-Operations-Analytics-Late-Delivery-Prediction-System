from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_operations_cleaned.csv"
OUTPUT = ROOT / "data" / "processed" / "dataco_orders.csv"
REPORT = ROOT / "reports" / "03_order_level_validation.txt"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "order_id",
        "order_item_total",
        "order_item_quantity",
        "order_profit_per_order",
        "delivery_status",
        "late_delivery_risk",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonne mancanti: {', '.join(missing)}")

    order_fields = [
        column
        for column in [
            "order_date",
            "shipping_date",
            "delivery_status",
            "late_delivery_risk",
            "days_for_shipping_real",
            "days_for_shipment_scheduled",
            "shipping_mode",
            "market",
            "order_region",
            "customer_segment",
            "order_status",
        ]
        if column in df.columns
    ]

    conflicts = {
        column: int(
            df.groupby("order_id")[column]
            .nunique(dropna=False)
            .gt(1)
            .sum()
        )
        for column in order_fields
    }

    profit_is_constant = (
        df.groupby("order_id")["order_profit_per_order"]
        .nunique(dropna=False)
        .le(1)
        .all()
    )

    aggregation = {
        "order_item_total": "sum",
        "order_item_quantity": "sum",
        "order_profit_per_order": "first" if profit_is_constant else "sum",
        **{column: "first" for column in order_fields},
    }

    if "order_item_cardprod_id" in df.columns:
        aggregation["order_item_cardprod_id"] = "nunique"

    if "category_name" in df.columns:
        aggregation["category_name"] = "nunique"

    orders = (
        df.groupby("order_id", as_index=False)
        .agg(aggregation)
        .rename(
            columns={
                "order_item_total": "revenue",
                "order_item_quantity": "quantity",
                "order_item_cardprod_id": "unique_products",
                "category_name": "unique_categories",
            }
        )
    )

    line_count = (
        df.groupby("order_id")
        .size()
        .rename("order_lines")
        .reset_index()
    )

    orders = orders.merge(line_count, on="order_id", how="left")

    orders["shipping_delay_days"] = (
        orders["days_for_shipping_real"]
        - orders["days_for_shipment_scheduled"]
    )

    orders["delivery_timing_group"] = orders["delivery_status"].replace(
        {
            "Late delivery": "Late",
            "Shipping on time": "On Time",
            "Advance shipping": "Early",
            "Shipping canceled": "Canceled",
        }
    )

    orders["loss_flag"] = (
        orders["order_profit_per_order"] < 0
    ).astype(int)

    orders.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    status_summary = (
        orders["delivery_timing_group"]
        .value_counts(dropna=False)
        .to_string()
    )

    conflict_text = "\n".join(
        f"{column}: {count:,} orders"
        for column, count in conflicts.items()
    )

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "03 - ORDER-LEVEL DATASET VALIDATION",
        "",
        f"Source rows: {len(df):,}",
        f"Unique orders: {len(orders):,}",
        f"Output columns: {orders.shape[1]}",
        f"Profit aggregation: {'first value per order' if profit_is_constant else 'sum of rows'}",
        "",
        "Order-level field conflicts:",
        conflict_text,
        "",
        "Delivery status summary:",
        status_summary,
        "",
        f"Output: {OUTPUT}",
    ]

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creato: {OUTPUT}")
    print(f"Creato: {REPORT}")
    print(f"Ordini: {len(orders):,} | Colonne: {orders.shape[1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)