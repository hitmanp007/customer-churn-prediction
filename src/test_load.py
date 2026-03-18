import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)
print("\nFirst 5 Rows:\n")
print(df.head())