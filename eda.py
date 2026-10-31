
# EXPLORATORY DATA ANALYSIS (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 2. LOAD DATASET

df = pd.read_csv("data.csv", sep=";")

# Clean column names

df.columns = df.columns.str.strip()

print("\nDATASET LOADED SUCCESSFULLY")
# 3. DATASET OVERVIEW
print("\n1. DATASET OVERVIEW")
print("-" * 55)
print("Dataset Shape:", df.shape)
print("Number of Records:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst Five Records:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe().T)

# 4. DATA QUALITY ANALYSIS

print("\n2. DATA QUALITY ANALYSIS")
print("-" * 55)

print("\nMissing Values:")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nData Types:")
print(df.dtypes.value_counts())

print("\nUnique Values in Each Column:")
print(df.nunique().sort_values(ascending=False))

# 5. TARGET VARIABLE ANALYSIS

print("\n3. STUDENT OUTCOME ANALYSIS")
print("-" * 55)

target_counts = df["Target"].value_counts()

print("\nStudent Outcome Counts:")
print(target_counts)

print("\nStudent Outcome Percentages:")

target_percentages = (
    df["Target"].value_counts(normalize=True) * 100
).round(2)

print(target_percentages)


# Visualization 1: Student Outcome Distribution

plt.figure(figsize=(10, 6))

ax = sns.countplot(
    data=df,
    x="Target",
    order=["Dropout", "Enrolled", "Graduate"]
)

plt.title(
    "Distribution of Academic Outcomes Among Students",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.xlabel("Academic Outcome", fontsize=11)
plt.ylabel("Number of Students", fontsize=11)

for container in ax.containers:
    ax.bar_label(container, padding=4, fontsize=10)

plt.tight_layout()
plt.show()

# 6. DEMOGRAPHIC ANALYSIS

print("\nAge Statistics:")
print(df["Age at enrollment"].describe())

print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nMarital Status Distribution:")
print(df["Marital status"].value_counts())


# Visualization 2: Age Distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Age at enrollment",
    bins=25,
    kde=True
)

plt.title(
    "Age Profile of Students at Enrollment",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.xlabel("Age at Enrollment", fontsize=11)
plt.ylabel("Number of Students", fontsize=11)

plt.tight_layout()
plt.show()

# 7. ACADEMIC PERFORMANCE ANALYSIS


academic_columns = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

print("\nAcademic Performance Summary:")
print(df[academic_columns].describe().T)


# Visualization 3: Academic Performance by Outcome

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Target",
    y="Curricular units 1st sem (grade)",
    order=["Dropout", "Enrolled", "Graduate"]
)

plt.title(
    "First-Semester Academic Performance Across Student Outcomes",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.xlabel("Academic Outcome", fontsize=11)
plt.ylabel("First-Semester Academic Grade", fontsize=11)

plt.tight_layout()
plt.show()

# 8. FINANCIAL ANALYSIS

financial_columns = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder"
]

for column in financial_columns:

    print("\n", column)
    print(df[column].value_counts())


# Financial Characteristics Across Student Outcomes

print("\nFinancial Characteristics Across Student Outcomes")
print("-" * 55)

for column in financial_columns:

    print("\n", column)

    financial_comparison = pd.crosstab(
        df["Target"],
        df[column],
        normalize="index"
    ) * 100

    print(financial_comparison.round(2))

# 9. ADMISSION ANALYSIS
admission_columns = [
    "Application mode",
    "Application order",
    "Course",
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade"
]

print("\nAdmission Statistics:")
print(df[admission_columns].describe().T)

# 10. ECONOMIC INDICATOR ANALYSIS


economic_columns = [
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

print("\nEconomic Indicator Summary:")
print(df[economic_columns].describe().T)

# 11. CORRELATION ANALYSIS


print("\n9. ACADEMIC CORRELATION ANALYSIS")
print("-" * 55)

correlation_columns = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Age at enrollment",
    "Admission grade"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))


# Visualization 4: Correlation Heatmap

plt.figure(figsize=(11, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title(
    "Correlation Analysis of Academic and Student Characteristics",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.tight_layout()
plt.show()


# 12. FINAL EDA SUMMARY

print("\nDataset Records:", len(df))
print("Dataset Features:", df.shape[1] - 1)
print("Target Classes:", df["Target"].nunique())

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())

print("\nStudent Outcome Distribution:")
print(target_counts)

print("\nStudent Outcome Percentages:")
print(target_percentages)

print("\nVisualizations Generated:")

print("1. Count Plot - Academic Outcome Distribution")
print("2. Histogram - Student Age Profile")
print("3. Box Plot - Academic Performance Across Outcomes")
print("4. Heatmap - Academic and Student Variable Correlations")

print("\nAdditional Analysis:")
print("Financial Characteristics Across Student Outcomes")



