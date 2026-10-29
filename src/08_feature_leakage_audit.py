

import pandas as pd

# 1. Load the engineered dataset
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "engineered_data.csv")

print("\n========== HERAI FEATURE LEAKAGE AUDIT ==========")

# 2. Check required columns
required_columns = [
    "user_id",
    "start_date",
    "overall_health_score",
    "next_health_score"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# 3. Check target alignment BEFORE dropping rows
df["start_date"] = pd.to_datetime(
    df["start_date"], errors="coerce"
)

df = df.sort_values(
    ["user_id", "start_date"]
).reset_index(drop=True)

expected_target = (
    df.groupby("user_id")["overall_health_score"].shift(-1)
)

valid_rows = df["next_health_score"].notna()

matches = (
    df.loc[valid_rows, "next_health_score"].round(8)
    == expected_target.loc[valid_rows].round(8)
)

print("\n1. Target alignment")
print("Rows checked:", int(valid_rows.sum()))
print("Matches:", int(matches.sum()))
print("Mismatches:", int((~matches).sum()))

# 4. Define model inputs
excluded_columns = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

X = X.loc[valid_rows]

print("\n2. Direct target leakage")
print("Target included in inputs:", "next_health_score" in X.columns)

# 5. Check all feature names for potential warning signs
print("\n3. Features needing manual review")

keywords = [
    "next", "future", "target", "outcome",
    "after", "next_cycle", "following"
]

flagged = [
    col for col in X.columns
    if any(word in col.lower() for word in keywords)
]

if flagged:
    for col in flagged:
        print("-", col)
else:
    print("No feature names matched the warning keywords.")

# 6. Check duplicate and constant features
print("\n4. Duplicate or constant feature checks")

constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

duplicate_columns = [
    col for col in X.columns
    if X[col].dropna().equals(
        df.loc[valid_rows, "next_health_score"].reset_index(drop=True)
    )
]

print("Constant columns:", constant_columns)
print("Features identical to target:", duplicate_columns)

# 7. Report the features being audited
print("\n5. Feature inventory")
print("Number of input features:", len(X.columns))

for col in X.columns:
    print("-", col)

print("\n========== AUDIT COMPLETE ==========")



print("\n========== FEATURE TIMING AUDIT ==========")

excluded_from_model = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number",
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period"
]

remaining_features = [
    col for col in df.columns
    if col not in excluded_from_model
]

print("\nFeatures in the 37-feature candidate:")
for i, feature in enumerate(remaining_features, start=1):
    print(f"{i}. {feature}")

print("\nTotal features:", len(remaining_features))
