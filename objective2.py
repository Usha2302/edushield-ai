
# OBJECTIVE 2: FEATURE ENGINEERING AND PREPROCESSING PREPARATION


# Research Question:
# How can meaningful academic progression and financial indicators
# be engineered and transformed to prepare the dataset for
# machine-learning prediction?

# Methods:
# - Feature engineering
# - Approval rates and unapproved units
# - Academic progression indicators
# - Financial indicator engineering
# - Categorical encoding preparation
# - Feature scaling preparation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

print("\nDATASET LOADED SUCCESSFULLY")
print("Original Dataset Shape:", df.shape)

original_shape = df.shape

# 3. INITIAL DATA VALIDATION


print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Target"].value_counts())

# 4. FEATURE ENGINEERING


# FIRST SEMESTER FEATURES

df["1st_Sem_Approval_Rate"] = np.divide(
    df["Curricular units 1st sem (approved)"],
    df["Curricular units 1st sem (enrolled)"],
    out=np.zeros(len(df), dtype=float),
    where=df["Curricular units 1st sem (enrolled)"] != 0
)

df["1st_Sem_Unapproved_Units"] = (
    df["Curricular units 1st sem (enrolled)"]
    - df["Curricular units 1st sem (approved)"]
)

df["1st_Sem_Evaluations_Per_Enrolled_Unit"] = np.divide(
    df["Curricular units 1st sem (evaluations)"],
    df["Curricular units 1st sem (enrolled)"],
    out=np.zeros(len(df), dtype=float),
    where=df["Curricular units 1st sem (enrolled)"] != 0
)

df["1st_Sem_No_Enrollment_Flag"] = (
    df["Curricular units 1st sem (enrolled)"] == 0
).astype(int)


# SECOND SEMESTER FEATURES

df["2nd_Sem_Approval_Rate"] = np.divide(
    df["Curricular units 2nd sem (approved)"],
    df["Curricular units 2nd sem (enrolled)"],
    out=np.zeros(len(df), dtype=float),
    where=df["Curricular units 2nd sem (enrolled)"] != 0
)

df["2nd_Sem_Unapproved_Units"] = (
    df["Curricular units 2nd sem (enrolled)"]
    - df["Curricular units 2nd sem (approved)"]
)

df["2nd_Sem_Evaluations_Per_Enrolled_Unit"] = np.divide(
    df["Curricular units 2nd sem (evaluations)"],
    df["Curricular units 2nd sem (enrolled)"],
    out=np.zeros(len(df), dtype=float),
    where=df["Curricular units 2nd sem (enrolled)"] != 0
)

df["2nd_Sem_No_Enrollment_Flag"] = (
    df["Curricular units 2nd sem (enrolled)"] == 0
).astype(int)


# ACADEMIC PROGRESSION FEATURES

df["Semester_Grade_Change"] = (
    df["Curricular units 2nd sem (grade)"]
    - df["Curricular units 1st sem (grade)"]
)

df["Semester_Approval_Change"] = (
    df["Curricular units 2nd sem (approved)"]
    - df["Curricular units 1st sem (approved)"]
)

df["Average_Semester_Grade"] = (
    df["Curricular units 1st sem (grade)"]
    + df["Curricular units 2nd sem (grade)"]
) / 2


# FINANCIAL FEATURES

df["Tuition_Overdue_Flag"] = (
    1 - df["Tuition fees up to date"]
)

df["Financial_Difficulty_Count"] = (
    df["Debtor"] + df["Tuition_Overdue_Flag"]
)


# LIST OF NEWLY ENGINEERED FEATURES

new_features = [
    "1st_Sem_Approval_Rate",
    "1st_Sem_Unapproved_Units",
    "1st_Sem_Evaluations_Per_Enrolled_Unit",
    "1st_Sem_No_Enrollment_Flag",
    "2nd_Sem_Approval_Rate",
    "2nd_Sem_Unapproved_Units",
    "2nd_Sem_Evaluations_Per_Enrolled_Unit",
    "2nd_Sem_No_Enrollment_Flag",
    "Semester_Grade_Change",
    "Semester_Approval_Change",
    "Average_Semester_Grade",
    "Tuition_Overdue_Flag",
    "Financial_Difficulty_Count"
]

print("\nNewly Engineered Features:")

for feature in new_features:
    print(feature)

# 5. ENGINEERED FEATURE SUMMARY

print(df[new_features].describe().round(4))


# 6. ENGINEERED FEATURE VALIDATION


# APPROVAL RATE VALIDATION

for feature in [
    "1st_Sem_Approval_Rate",
    "2nd_Sem_Approval_Rate"
]:

    print("\n", feature)
    print("Minimum:", df[feature].min())
    print("Maximum:", df[feature].max())

    outside_range = (
        (df[feature] < 0) | (df[feature] > 1)
    ).sum()

    print("Values Outside 0-1:", outside_range)


# EVALUATION FEATURE VALIDATION

evaluation_features = [
    "1st_Sem_Evaluations_Per_Enrolled_Unit",
    "2nd_Sem_Evaluations_Per_Enrolled_Unit"
]

print("\nEvaluation Activity Validation:")

for feature in evaluation_features:
    print(
        feature,
        "- Minimum:", round(df[feature].min(), 3),
        "| Maximum:", round(df[feature].max(), 3)
    )

print(
    "\nNote: Evaluation activity can exceed 1 because "
    "multiple evaluations may occur per enrolled unit."
)


# FINANCIAL INDICATOR VALIDATION

print("\nFinancial Indicator Validation:")

print(
    df[
        [
            "Debtor",
            "Tuition fees up to date",
            "Tuition_Overdue_Flag",
            "Financial_Difficulty_Count"
        ]
    ].head()
)

print("\nFinancial Difficulty Count Distribution:")

print(
    df["Financial_Difficulty_Count"]
    .value_counts()
    .sort_index()
)


# ACADEMIC FEATURE VALIDATION

print("\nAcademic Feature Validation:")

print(
    "Negative First-Semester Unapproved Units:",
    (df["1st_Sem_Unapproved_Units"] < 0).sum()
)

print(
    "Negative Second-Semester Unapproved Units:",
    (df["2nd_Sem_Unapproved_Units"] < 0).sum()
)


# 7. VISUALIZATION: SCATTER PLOT


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="1st_Sem_Approval_Rate",
    y="Curricular units 1st sem (grade)",
    hue="Target",
    alpha=0.55,
    s=55,
    edgecolor=None
)

plt.title(
    "Academic Performance and Approval Rate Across Student Outcomes",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("First-Semester Approval Rate")
plt.ylabel("First-Semester Academic Grade")

plt.legend(title="Student Outcome")
plt.tight_layout()
plt.show()


# 8. FEATURE TYPE IDENTIFICATION


categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "1st_Sem_No_Enrollment_Flag",
    "2nd_Sem_No_Enrollment_Flag",
    "Tuition_Overdue_Flag"
]


# REMOVE TARGET FROM INPUT FEATURES

X = df.drop(columns=["Target"])
y = df["Target"]


# IDENTIFY NUMERICAL FEATURES

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

print("\nNumerical Feature Count:", len(numerical_features))
print("Categorical Feature Count:", len(categorical_features))

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nTarget Variable:", "Target")

print("\nTarget Classes:")
print(y.unique())


# 9. PREPROCESSING PREPARATION


print("\n" + "=" * 75)
print("7. PREPROCESSING PREPARATION")
print("=" * 75)

print("\nNumerical Features:")
print("StandardScaler will be applied.")

print("\nCategorical Features:")
print("One-hot encoding will be applied to nominal variables.")

print("\nTarget Variable:")
print("LabelEncoder will be used if numerical target labels are required.")

print("\nPreprocessing Plan:")
print("1. Split data into training and testing sets.")
print("2. Fit encoding and scaling using training data only.")
print("3. Apply the fitted transformations to testing data.")
print("4. Exclude Target from all input features.")
print("5. Select features according to prediction timing.")


# 10. FINAL FEATURE ENGINEERING SUMMARY


print("Original Dataset Shape:", original_shape)
print("Dataset Shape After Feature Engineering:", df.shape)

print("Original Input Features:", original_shape[1] - 1)
print("Newly Engineered Features:", len(new_features))
print("Total Input Features:", X.shape[1])

print("\nMissing Values After Feature Engineering:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print(
    "\nTarget Variable Excluded from Input Features:",
    "Target" not in X.columns
)

print("\nFeature engineering and preprocessing preparation completed.")

