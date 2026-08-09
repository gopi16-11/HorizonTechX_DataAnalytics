import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# TASK 3: DATA VISUALIZATION
# Dataset: Titanic_100.csv
# ==========================================

print("========== TASK 3: DATA VISUALIZATION ==========\n")

# Load dataset
df = pd.read_csv("Titanic_100.csv")

print("Dataset loaded successfully!\n")

# Display first 5 records
print("----- FIRST 5 RECORDS -----")
print(df.head())

# ==========================================
# 1. SURVIVAL COUNT
# ==========================================

survival_count = df["Survived"].value_counts()

plt.figure(figsize=(7, 5))

survival_count.plot(kind="bar")

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("survival_count.png")
plt.show()


# ==========================================
# 2. PASSENGERS BY CLASS
# ==========================================

class_count = df["Pclass"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

class_count.plot(kind="bar")

plt.title("Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("passengers_by_class.png")
plt.show()


# ==========================================
# 3. AGE DISTRIBUTION
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
plt.savefig("age_distribution.png")
plt.show()


# ==========================================
# 4. SURVIVAL BY GENDER
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
plt.savefig("survival_by_gender.png")
plt.show()


# ==========================================
# 5. FARE DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

df["Fare"].plot(
    kind="hist",
    bins=30
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.show()


# ==========================================
# FINAL OUTPUT
# ==========================================

print("----- VISUALIZATION SUMMARY -----")
print("Total Passengers:", len(df))
print("Total Survived:", df["Survived"].sum())
print("Total Not Survived:", (df["Survived"] == 0).sum())

print("\nAll visualizations created successfully!")
print("Task 3 completed successfully!")