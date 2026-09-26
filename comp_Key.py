import pandas as pd

# Load dataset
df = pd.read_csv("out/final_dataset_features.csv")

# Create a composite key
df["time_key"] = (
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2) + "-" +
    df["weekday"].astype(str) + "-" +
    df["hour"].astype(str).str.zfill(2)
)

# Save updated dataset
df.to_csv("compositeKeyDataset.csv", index=False)
