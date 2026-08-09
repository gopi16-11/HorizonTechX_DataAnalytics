import pandas as pd

# Load Titanic dataset
df = pd.read_csv("Titanic_100.csv")

print("========== TITANIC DATASET ==========")

# Display first 10 rows
print("\nFirst 10 Records:")
print(df.head(10))

# Dataset size
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Survival count
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# Gender distribution
print("\nGender Distribution:")
print(df["Sex"].value_counts())

print("\n========== TASK 1 COMPLETED ==========")

