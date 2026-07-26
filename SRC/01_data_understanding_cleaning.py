

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "dataco_operations_analytics.csv"
OUTPUT = ROOT / "data" / "processed" / "dataco_operations_cleaned.csv"
REPORT = ROOT / "reports" / "01_data_understanding_report.txt"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)
    rows_before = len(df)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    df = df.rename(
        columns={
            "days_for_shipping": "days_for_shipping_real",
            "days_for_shipment": "days_for_shipment_scheduled",
        }
    )

    for column in df.select_dtypes(include="object"):
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    for column in ("order_date", "shipping_date"):
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    numeric_columns = [
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "late_delivery_risk",
        "order_item_total",
        "sales",
        "order_profit_per_order",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    shipping_columns = {
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
    }

    if shipping_columns.issubset(df.columns):
        df["shipping_delay_days"] = (
            df["days_for_shipping_real"]
            - df["days_for_shipment_scheduled"]
        )

        df["delivery_timing_group"] = "On Time"
        df.loc[
            df["shipping_delay_days"] > 0,
            "delivery_timing_group",
        ] = "Late"
        df.loc[
            df["shipping_delay_days"] < 0,
            "delivery_timing_group",
        ] = "Early"

    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = (
            df["order_date"]
            .dt.to_period("M")
            .astype("string")
        )
        df["order_day_of_week"] = (
            df["order_date"]
            .dt.day_name()
        )

    if "order_item_total" in df.columns:
        df["revenue_value"] = df["order_item_total"]
    elif "sales" in df.columns:
        df["revenue_value"] = df["sales"]

    if "order_profit_per_order" in df.columns:
        df["profit_status"] = "Profit"
        df.loc[
            df["order_profit_per_order"] < 0,
            "profit_status",
        ] = "Loss"

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    missing = df.isna().sum()

    report = [
        "E-COMMERCE OPERATIONS ANALYTICS & DECISION SYSTEM",
        "01 - DATA UNDERSTANDING AND CLEANING",
        "",
        f"Input: {INPUT}",
        f"Output: {OUTPUT}",
        f"Rows before: {rows_before:,}",
        f"Rows after: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Exact duplicates removed: {duplicates:,}",
        f"Missing cells: {int(missing.sum()):,}",
        "",
        "Columns with missing values:",
        missing[missing > 0]
        .sort_values(ascending=False)
        .to_string(),
        "",
        "Modeling warning:",
        "Do not use delivery_status, days_for_shipping_real or shipping_date",
        "as predictors of late_delivery_risk because they create data leakage.",
    ]

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Creato: {OUTPUT}")
    print(f"Creato: {REPORT}")
    print(
        f"Righe: {len(df):,} | "
        f"Colonne: {df.shape[1]} | "
        f"Duplicati rimossi: {duplicates:,}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)
