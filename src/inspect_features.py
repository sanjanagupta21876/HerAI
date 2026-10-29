
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "Period_Log.csv")

df = df.sort_values(
    ["user_id", "start_date"]
).reset_index(drop=True)

print("Preparation flag by cycle:")
print(
    df[
        [
            "user_id",
            "cycle_number",
            "start_date",
            "prepared_before_period",
            "overall_health_score"
        ]
    ].head(20).to_string(index=False)
)

print("\nPreparation flag by cycle number:")
print(
    df.groupby("cycle_number")["prepared_before_period"]
    .mean()
    .head(15)
)



print("\nHealth score by preparation flag:")
print(
    df.groupby("prepared_before_period")["overall_health_score"]
      .agg(["count", "mean", "std"])
)

print("\nPreparation flag correlation with current health score:")
print(
    df["prepared_before_period"].corr(
        df["overall_health_score"]
    )
)

