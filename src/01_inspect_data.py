import pandas as pd

# Load the raw datasets
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
period = pd.read_csv(ROOT / "data" / "raw" / "Period_Log.csv")
profile = pd.read_csv(ROOT / "data" / "raw" / "User_Profile.csv")

period["start_date"] = pd.to_datetime(
    period["start_date"],
    errors="coerce"
)

print("========== PERIOD LOG ==========")

print("Shape:", period.shape)

print("\nColumns:")
print(period.columns.tolist())

print("\nData Types:")
print(period.dtypes)

print("\nMissing Values:")
print(period.isnull().sum())


print("\n\n========== USER PROFILE ==========")

print("Shape:", profile.shape)

print("\nColumns:")
print(profile.columns.tolist())

print("\nData Types:")
print(profile.dtypes)

print("\nMissing Values:")
print(profile.isnull().sum())


print("\n========== DUPLICATE CHECK ==========")

# Exact duplicate rows
print("\nPeriod Log exact duplicates:")
print(period.duplicated().sum())

print("\nUser Profile exact duplicates:")
print(profile.duplicated().sum())


# Duplicate user + cycle combinations
print("\nDuplicate user + cycle combinations:")
print(
    period.duplicated(
        subset=["user_id", "cycle_number"]
    ).sum()
)


# Duplicate user IDs in profile
print("\nDuplicate user IDs in User Profile:")
print(
    profile.duplicated(
        subset=["user_id"]
    ).sum()
)

print("\n========== NUMERICAL VALUE CHECK ==========")

print("\n--- Period Log ---")
print(period.describe().T)

print("\n--- User Profile ---")
print(profile.describe().T)

print("\n========== CATEGORICAL VALUE CHECK ==========")

print("\n--- Period Log ---")

for col in period.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(period[col].value_counts(dropna=False))


print("\n--- User Profile ---")

for col in profile.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(profile[col].value_counts(dropna=False))

print("\n========== DATE CHECK ==========")

print("Date type:")
print(period["start_date"].dtype)

print("\nInvalid dates:")
print(period["start_date"].isna().sum())

print("\nEarliest date:")
print(period["start_date"].min())

print("\nLatest date:")
print(period["start_date"].max())

print("\n========== CYCLES PER USER ==========")

cycles_per_user = period.groupby("user_id")["cycle_number"].count()

print("\nNumber of users:")
print(cycles_per_user.size)

print("\nMinimum cycles for one user:")
print(cycles_per_user.min())

print("\nMaximum cycles for one user:")
print(cycles_per_user.max())

print("\nAverage cycles per user:")
print(cycles_per_user.mean())

print("\nCycle count distribution:")
print(cycles_per_user.value_counts().sort_index())

print("\n========== HEALTH SCORE CHECK ==========")

health = period["overall_health_score"]

print("\nNumber of values:")
print(health.count())

print("\nNumber of unique scores:")
print(health.nunique())

print("\nMinimum score:")
print(health.min())

print("\nMaximum score:")
print(health.max())

print("\nAverage score:")
print(health.mean())

print("\nMedian score:")
print(health.median())

print("\nScore distribution:")
print(health.value_counts().sort_index())

print("\n========== HEALTH SCORE CORRELATIONS ==========")

health_correlations = period.select_dtypes(
    include="number"
).corr()["overall_health_score"].sort_values(
    ascending=False
)

print(health_correlations)

print("\n========== PAIN VS HEALTH SCORE ==========")

print(
    period.groupby("pain_level")["overall_health_score"]
    .agg(["count", "mean", "min", "max"])
)
print("\n========== WORK HOURS LOST VS HEALTH SCORE ==========")

print(
    period.groupby("work_hours_lost")["overall_health_score"]
    .agg(["count", "mean", "min", "max"])
)
print("\n========== STRESS VS HEALTH SCORE ==========")

print(
    period.groupby("stress_score_cycle")["overall_health_score"]
    .agg(["count", "mean", "min", "max"])
)
print("\n========== HEALTH SCORE FORMULA CHECK ==========")

features = [
    "pain_level",
    "work_hours_lost",
    "stress_score_cycle",
    "concentration_score",
    "energy_level",
    "mood_score",
    "sleep_hours_cycle"
]

temp = period[features + ["overall_health_score"]].dropna()

X = temp[features]
y = temp["overall_health_score"]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

r2 = r2_score(y, predictions)

print("R²:", r2)

print("\nCoefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)

print("\n========== NEXT CYCLE TARGET CHECK ==========")

period_sorted = period.sort_values(
    ["user_id", "start_date"]
).copy()

period_sorted["next_health_score"] = (
    period_sorted
    .groupby("user_id")["overall_health_score"]
    .shift(-1)
)

print("\nTotal rows:")
print(len(period_sorted))

print("\nNext-cycle target available:")
print(period_sorted["next_health_score"].notna().sum())

print("\nNext-cycle target missing:")
print(period_sorted["next_health_score"].isna().sum())

print("\nExample:")
print(
    period_sorted[
        ["user_id", "cycle_number",
         "overall_health_score",
         "next_health_score"]
    ].head(15)
)

print("\n========== PREVIOUS VS NEXT HEALTH SCORE ==========")

valid = period_sorted[
    ["overall_health_score", "next_health_score"]
].dropna()

print(
    valid.corr()
)
print("\n========== PREVIOUS HEALTH SCORE ==========")

period_sorted["previous_health_score"] = (
    period_sorted
    .groupby("user_id")["overall_health_score"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "overall_health_score",
            "previous_health_score",
            "next_health_score"
        ]
    ].head(15)
)
print("\n========== PREVIOUS CYCLE LENGTH ==========")

period_sorted["previous_cycle_length_feature"] = (
    period_sorted
    .groupby("user_id")["cycle_length_days"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "cycle_length_days",
            "previous_cycle_length_feature"
        ]
    ].head(15)
)
print("\n========== PREVIOUS PAIN LEVEL ==========")

period_sorted["previous_pain_level"] = (
    period_sorted
    .groupby("user_id")["pain_level"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "pain_level",
            "previous_pain_level"
        ]
    ].head(15)
)
print("\n========== PREVIOUS STRESS SCORE ==========")

period_sorted["previous_stress_score"] = (
    period_sorted
    .groupby("user_id")["stress_score_cycle"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "stress_score_cycle",
            "previous_stress_score"
        ]
    ].head(15)
)
print("\n========== PREVIOUS SLEEP HOURS ==========")

period_sorted["previous_sleep_hours"] = (
    period_sorted
    .groupby("user_id")["sleep_hours_cycle"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "sleep_hours_cycle",
            "previous_sleep_hours"
        ]
    ].head(15)
)
print("\n========== PREVIOUS MOOD SCORE ==========")

period_sorted["previous_mood_score"] = (
    period_sorted
    .groupby("user_id")["mood_score"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "mood_score",
            "previous_mood_score"
        ]
    ].head(15)
)
print("\n========== PREVIOUS ENERGY LEVEL ==========")

period_sorted["previous_energy_level"] = (
    period_sorted
    .groupby("user_id")["energy_level"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "energy_level",
            "previous_energy_level"
        ]
    ].head(15)
)
print("\n========== PREVIOUS CONCENTRATION SCORE ==========")

period_sorted["previous_concentration_score"] = (
    period_sorted
    .groupby("user_id")["concentration_score"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "concentration_score",
            "previous_concentration_score"
        ]
    ].head(15)
)

print("\n========== PREVIOUS WORK HOURS LOST ==========")

period_sorted["previous_work_hours_lost"] = (
    period_sorted
    .groupby("user_id")["work_hours_lost"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "work_hours_lost",
            "previous_work_hours_lost"
        ]
    ].head(15)
)
print("\n========== PREVIOUS FLOW LEVEL ==========")

period_sorted["previous_flow_level"] = (
    period_sorted
    .groupby("user_id")["flow_level"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "flow_level",
            "previous_flow_level"
        ]
    ].head(15)
)
print("\n========== PREVIOUS PMS SYMPTOMS ==========")

period_sorted["previous_pms_symptoms"] = (
    period_sorted
    .groupby("user_id")["pms_symptoms"]
    .shift(1)
)

print(
    period_sorted[
        [
            "user_id",
            "cycle_number",
            "pms_symptoms",
            "previous_pms_symptoms"
        ]
    ].head(15)
)

print("\n========== HISTORICAL FEATURE MISSING VALUES ==========")

historical_features = [
    "previous_health_score",
    "previous_cycle_length_feature",
    "previous_pain_level",
    "previous_stress_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_energy_level",
    "previous_concentration_score",
    "previous_work_hours_lost",
    "previous_flow_level",
    "previous_pms_symptoms"
]

print(
    period_sorted[historical_features].isna().sum()
)

print("\n========== HISTORICAL FEATURES VS NEXT HEALTH ==========")

features_to_check = [
    "previous_health_score",
    "previous_cycle_length_feature",
    "previous_pain_level",
    "previous_stress_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_energy_level",
    "previous_concentration_score",
    "previous_work_hours_lost"
]

correlations = period_sorted[
    features_to_check + ["next_health_score"]
].corr()["next_health_score"].sort_values(
    ascending=False
)

print(correlations)