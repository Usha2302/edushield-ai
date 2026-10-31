
# ============================================================
# EDUSHIELD AI
# OBJECTIVE 5 — STUDENT RETENTION PRIORITIZATION
# ============================================================

# Research Question:
# How can predicted student outcomes and identified risk factors help
# institutions prioritize students for appropriate academic and financial support?
#
# Methods:
# Risk categorization, student-support prioritization and
# evidence-based recommendations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)



# 2. LOAD DATASET

df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()


print("\nDataset Shape:", df.shape)



# 3. FEATURE ENGINEERING


df["1st_Sem_Approval_Rate"] = np.where(
    df["Curricular units 1st sem (enrolled)"] > 0,
    df["Curricular units 1st sem (approved)"] /
    df["Curricular units 1st sem (enrolled)"],
    0
)

df["1st_Sem_Unapproved_Units"] = (
    df["Curricular units 1st sem (enrolled)"] -
    df["Curricular units 1st sem (approved)"]
)

df["1st_Sem_Evaluations_Per_Enrolled_Unit"] = np.where(
    df["Curricular units 1st sem (enrolled)"] > 0,
    df["Curricular units 1st sem (evaluations)"] /
    df["Curricular units 1st sem (enrolled)"],
    0
)

df["1st_Sem_No_Enrollment_Flag"] = (
    df["Curricular units 1st sem (enrolled)"] == 0
).astype(int)

df["Tuition_Overdue_Flag"] = (
    df["Tuition fees up to date"] == 0
).astype(int)

df["Financial_Difficulty_Count"] = (
    (df["Debtor"] == 1).astype(int) +
    df["Tuition_Overdue_Flag"]
)

print("\nFeature engineering completed.")


# 4. PREPARE INPUT FEATURES AND TARGET


y = df["Target"].map({
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
})

if y.isnull().any():
    raise ValueError("Unexpected target value found in dataset.")

# Exclude target and second-semester information

second_semester_columns = [
    col for col in df.columns
    if col.startswith("Curricular units 2nd sem")
    or col.startswith("2nd_Sem_")
]

X = df.drop(
    columns=["Target"] + second_semester_columns
)

# Categorical features

categorical_columns = [
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
    "1st_Sem_No_Enrollment_Flag"
]

categorical_columns = [
    col for col in categorical_columns
    if col in X.columns
]

print("\nNumber of Input Features:", X.shape[1])
print("Categorical Features:", len(categorical_columns))



# 5. TRAIN-TEST SPLIT

record_indices = df.index

(
    X_train,
    X_test,
    y_train,
    y_test,
    train_indices,
    test_indices
) = train_test_split(
    X,
    y,
    record_indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 6. ENCODING AND FEATURE SCALING

# One-hot encoding

X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    dtype=int
)

# Align test features with training features

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 7. TRAIN LOGISTIC REGRESSION MODEL


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nMODEL EVALUATION")
print("-" * 65)

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "Macro F1-score:",
    round(f1_score(y_test, y_pred, average="macro"), 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Dropout", "Enrolled", "Graduate"],
        zero_division=0
    )
)


# 8. CALCULATE DROPOUT PROBABILITIES


dropout_class_index = list(model.classes_).index(0)

dropout_probabilities = model.predict_proba(
    X_test_scaled
)[:, dropout_class_index]

priority_df = df.loc[test_indices].copy()

priority_df["Predicted_Dropout_Probability"] = (
    dropout_probabilities
)

priority_df = priority_df.sort_values(
    by="Predicted_Dropout_Probability",
    ascending=False
)

priority_df["Risk_Rank"] = np.arange(
    1,
    len(priority_df) + 1
)


# 9. RISK PRIORITIZATION


total_students = len(priority_df)

high_cutoff = int(np.ceil(total_students * 0.20))
medium_cutoff = int(np.ceil(total_students * 0.50))

priority_df["Priority_Level"] = np.where(
    priority_df["Risk_Rank"] <= high_cutoff,
    "High",
    np.where(
        priority_df["Risk_Rank"] <= medium_cutoff,
        "Medium",
        "Low"
    )
)

print("\nRisk prioritization completed.")


# 10. SUPPORT RECOMMENDATIONS


def recommend_support(row):

    recommendations = []

    if (
        row["1st_Sem_Approval_Rate"] < 0.50
        or row["1st_Sem_No_Enrollment_Flag"] == 1
        or row["1st_Sem_Unapproved_Units"] > 0
    ):
        recommendations.append(
            "Academic counselling and subject support"
        )

    if (
        row["Debtor"] == 1
        or row["Tuition fees up to date"] == 0
    ):
        recommendations.append(
            "Financial counselling and scholarship guidance"
        )

    if not recommendations:
        recommendations.append(
            "Regular academic progress monitoring"
        )

    return "; ".join(recommendations)


priority_df["Recommended_Support"] = priority_df.apply(
    recommend_support,
    axis=1
)



# 11. PRIORITIZATION SUMMARY


print("\nSTUDENT PRIORITIZATION SUMMARY")
print("-" * 65)

priority_counts = (
    priority_df["Priority_Level"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

print(priority_counts)

print("\nAverage Predicted Dropout Probability by Priority:")

average_probabilities = (
    priority_df.groupby("Priority_Level")[
        "Predicted_Dropout_Probability"
    ]
    .mean()
    .reindex(["High", "Medium", "Low"])
)

print(average_probabilities.round(4))


# 12. EVALUATE HIGH-PRIORITY GROUP


# Actual outcomes are used only for retrospective evaluation.

actual_dropout = priority_df["Target"] == "Dropout"

high_priority = priority_df["Priority_Level"] == "High"

actual_dropout_count = actual_dropout.sum()
high_priority_count = high_priority.sum()

captured_dropouts = (
    actual_dropout & high_priority
).sum()

dropout_capture_rate = (
    captured_dropouts / actual_dropout_count
    if actual_dropout_count > 0 else 0
)

high_priority_dropout_precision = (
    captured_dropouts / high_priority_count
    if high_priority_count > 0 else 0
)

overall_dropout_rate = actual_dropout.mean()

lift = (
    high_priority_dropout_precision / overall_dropout_rate
    if overall_dropout_rate > 0 else 0
)

print("\nHIGH-PRIORITY GROUP EVALUATION")
print("-" * 65)

print("Actual Dropouts in Test Set:", actual_dropout_count)
print("Students in High-Priority Group:", high_priority_count)
print("Actual Dropouts Captured:", captured_dropouts)

print(
    "Dropout Capture Rate:",
    round(dropout_capture_rate, 4)
)

print(
    "High-Priority Dropout Precision:",
    round(high_priority_dropout_precision, 4)
)

print(
    "Overall Dropout Rate:",
    round(overall_dropout_rate, 4)
)

print("Prioritization Lift:", round(lift, 4))



# 13. VISUALIZATION 1 — PIE CHART


plt.figure(figsize=(8, 6))

priority_labels = [
    "High Priority",
    "Medium Priority",
    "Low Priority"
]

priority_values = [
    priority_counts["High"],
    priority_counts["Medium"],
    priority_counts["Low"]
]

plt.pie(
    priority_values,
    labels=priority_labels,
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.08, 0, 0),
    wedgeprops={"edgecolor": "white"}
)

plt.title(
    "Distribution of Students Across Retention Priority Groups",
    fontsize=13,
    fontweight="bold",
    pad=18
)

plt.tight_layout()


plt.show()

# 14. VISUALIZATION 2 — COLUMN CHART


metric_names = [
    "Overall Dropout Rate",
    "High-Priority Dropout Precision",
    "Dropout Capture Rate"
]

metric_values = [
    overall_dropout_rate * 100,
    high_priority_dropout_precision * 100,
    dropout_capture_rate * 100
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    metric_names,
    metric_values,
    width=0.55
)

plt.title(
    "Effectiveness of High-Priority Student Retention Screening",
    fontsize=13,
    fontweight="bold",
    pad=18
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Percentage (%)")

plt.ylim(0, 100)
plt.xticks(rotation=12, ha="right")

for bar, value in zip(bars, metric_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()


plt.show()


# 15. EXPORT PRIORITIZED STUDENT RECORDS


# Actual Target is deliberately excluded from the operational export.

export_columns = [
    "Risk_Rank",
    "Predicted_Dropout_Probability",
    "Priority_Level",
    "Recommended_Support"
]

priority_df[export_columns].to_csv(
    "student_support_priority.csv",
    index=True,
    index_label="Student_Record_ID"
)

print("\nExported: student_support_priority.csv")


# 16. TOP 10 STUDENTS FOR SUPPORT


print("\nTOP 10 STUDENTS FOR SUPPORT")
print("-" * 65)

print(
    priority_df[export_columns]
    .head(10)
    .to_string()
)

# 17. FINAL SUMMARY


print("\nVisualizations Generated:")
print("1. Retention Priority Distribution")
print("2. High-Priority Screening Effectiveness")

print("\nSaved Files:")
print("1. Objective5_Retention_Priority_Distribution.png")
print("2. Objective5_High_Priority_Screening_Effectiveness.png")
print("3. student_support_priority.csv")

print("\nRisk Group Distribution:")
print(priority_counts)

print("\nOBJECTIVE 5 COMPLETED SUCCESSFULLY.")
print("=" * 65)

