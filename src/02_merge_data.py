
import pandas as pd
from pathlib import Path

print("\n========== HERAI DATA MERGING ==========")

# Get the main project directory
ROOT = Path(__file__).resolve().parent.parent

# Define paths to the raw datasets
period_path = ROOT / "data" / "raw" / "Period_Log.csv"
profile_path = ROOT / "data" / "raw" / "User_Profile.csv"

# Load both datasets
period = pd.read_csv(period_path)
profile = pd.read_csv(profile_path)

print("\nOriginal dataset shapes:")
print("Period log:", period.shape)
print("User profile:", profile.shape)

# Check the merging key
print("\nMissing user IDs:")
print("Period log:", period["user_id"].isna().sum())
print("User profile:", profile["user_id"].isna().sum())

# Each user should have one profile record
if profile["user_id"].duplicated().any():
    raise ValueError(
        "Duplicate user IDs found in User_Profile.csv"
    )

# Merge each user's profile with their cycle records
merged_data = period.merge(
    profile,
    on="user_id",
    how="left",
    validate="many_to_one"
)

print("\n========== MERGE RESULTS ==========")
print("Merged dataset shape:", merged_data.shape)
print(
    "Missing user IDs after merge:",
    merged_data["user_id"].isna().sum()
)

# Check whether profile data was matched
profile_columns = [
    "age",
    "bmi",
    "diet_quality",
    "exercise_frequency",
    "stress_score_baseline"
]

available_columns = [
    col for col in profile_columns
    if col in merged_data.columns
]

print("\nMissing profile values after merge:")
print(merged_data[available_columns].isna().sum())

# Save the merged dataset
output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "merged_data.csv"
merged_data.to_csv(output_path, index=False)

print("\nMerged dataset saved successfully!")
print("Path:", output_path)
