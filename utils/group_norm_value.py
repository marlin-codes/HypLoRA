import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='Path to token_statistics_sorted.csv')
args = parser.parse_args()

# Load CSV
df = pd.read_csv(args.csv, names=["Token", "Frequency", "Average Norm"])

# Drop duplicate tokens
df = df.drop_duplicates(subset="Token", keep="first")

# Convert numeric fields
df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")
df["Average Norm"] = pd.to_numeric(df["Average Norm"], errors="coerce")

# Define token groups
groups = {
    "Group 1": ["to", "in", "have", "that", "and", "is", "for"],
    "Group 2": ["how", "much", "many", "time", "cost"],
    "Group 3": ["animal", "fruit", "number", "color", "size"],
    "Group 4": ["dog", "cow", "apple", "banana", "380", "480", "purple", "red", "medium", "small", "large"],
}

# Prepare result rows
results = []

# Analyze each group
for group_name, tokens in groups.items():
    group_df = df[df["Token"].isin(tokens)]
    if not group_df.empty:
        freq_mean = group_df["Frequency"].mean()
        freq_min = group_df["Frequency"].min()
        freq_max = group_df["Frequency"].max()

        norm_mean = group_df["Average Norm"].mean()
        norm_min = group_df["Average Norm"].min()
        norm_max = group_df["Average Norm"].max()

        results.append({
            "Group": group_name,
            "Frequency (Mean [Min∼Max])": f"{freq_mean:.1f} [{int(freq_min)} ∼ {int(freq_max)}]",
            "Norm (Mean [Min∼Max])": f"{norm_mean:.3f} [{norm_min:.3f} ∼ {norm_max:.3f}]"
        })
    else:
        results.append({
            "Group": group_name,
            "Frequency (Mean [Min∼Max])": "N/A",
            "Norm (Mean [Min∼Max])": "N/A"
        })

# Convert to DataFrame for display
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))