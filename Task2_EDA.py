import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Titanic.csv
# ==========================================

print("========== TASK 2: EDA ==========\n")

# 1. Load dataset
df = pd.read_csv("Titanic_100.csv")

print("Dataset loaded successfully!\n")

# 2. Display first 10 records
print("----- FIRST 10 RECORDS -----")
print(df.head(10))

# 3. Dataset shape
print("\n----- DATASET SHAPE -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. Column names
print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())

# 5. Data types
print("\n----- DATA TYPES -----")
print(df.dtypes)

# 6. Missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# 7. Statistical summary
print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())

# 8. Survival analysis
print("\n----- SURVIVAL COUNT -----")
print(df["Survived"].value_counts())

# 9. Gender analysis
print("\n----- GENDER COUNT -----")
print(df["Sex"].value_counts())

# 10. Average age
print("\n----- AVERAGE AGE -----")
print("Average Age:", round(df["Age"].mean(), 2))

# 11. Average fare
print("\n----- AVERAGE FARE -----")
print("Average Fare:", round(df["Fare"].mean(), 2))

# 12. Survival percentage
survival_percentage = df["Survived"].mean() * 100

print("\n----- SURVIVAL PERCENTAGE -----")
print("Survival Rate:", round(survival_percentage, 2), "%")

# ==========================================
# VISUALIZATION 1 - SURVIVAL COUNT
# ==========================================

plt.figure(figsize=(7, 5))

df["Survived"].value_counts().plot(
    kind="bar"
)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================
# VISUALIZATION 2 - AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

df["Age"].dropna().plot(
    kind="hist",
    bins=20
)

plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# ==========================================
# VISUALIZATION 3 - SURVIVAL BY GENDER
# ==========================================

gender_survival = pd.crosstab(
    df["Sex"],
    df["Survived"]
)

gender_survival.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Not Survived", "Survived"])
plt.tight_layout()
plt.show()

# ==========================================
# FINAL RESULT
# ==========================================

print("\n===================================")
print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("===================================")