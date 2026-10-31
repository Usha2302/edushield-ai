
# OBJECTIVE 4 — PREDICTION TIMING AND MODEL INTERPRETATION


# Research Question:
# How does the availability of academic information at different
# stages of a student's academic journey influence outcome prediction,
# and which factors influence the predictions?

# Methods:
# - Admission-time prediction
# - Admission + first-semester prediction
# - Logistic Regression
# - Performance comparison
# - Coefficient-based feature importance


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)





# 2. LOAD DATASET


df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

print("\nDATASET SHAPE:", df.shape)


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


# 4. DEFINE TARGET


target_mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

y = df["Target"].map(target_mapping)

if y.isnull().any():
    raise ValueError("Unexpected target value found in dataset.")

print("\nTarget Mapping:")
print(target_mapping)


# 5. DEFINE FEATURE GROUPS


second_sem_cols = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

first_sem_cols = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "1st_Sem_Approval_Rate",
    "1st_Sem_Unapproved_Units",
    "1st_Sem_Evaluations_Per_Enrolled_Unit",
    "1st_Sem_No_Enrollment_Flag"
]

# Admission-time features:
# Excludes first-semester and second-semester information.

admission_features = df.drop(
    columns=["Target"] + first_sem_cols + second_sem_cols
)

# Admission + first-semester features:
# Excludes all second-semester information.

first_sem_features = df.drop(
    columns=["Target"] + second_sem_cols
)

print("\nADMISSION-TIME FEATURES:",
      admission_features.shape[1])

print(
    "ADMISSION + FIRST-SEMESTER FEATURES:",
    first_sem_features.shape[1]
)

# 6. SAME TRAIN-TEST SPLIT FOR BOTH SCENARIOS

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("\nTraining Records:", len(train_indices))
print("Testing Records:", len(test_indices))



# 7. CATEGORICAL FEATURES


categorical_cols = [
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


# 8. TRAINING AND EVALUATION FUNCTION

def evaluate_scenario(X, scenario_name):

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    scenario_categorical = [
        col for col in categorical_cols
        if col in X_train.columns
    ]

    # One-hot encoding

    X_train = pd.get_dummies(
        X_train,
        columns=scenario_categorical,
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=scenario_categorical,
        dtype=int
    )

    # Align test columns with training columns

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Feature scaling

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    dropout_recall = recall_score(
        y_test,
        y_pred,
        labels=[0],
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 65)
    print(scenario_name)
    print("=" * 65)

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Dropout Recall:", round(dropout_recall, 4))

    result = {
        "Scenario": scenario_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Dropout Recall": dropout_recall
    }

    return result, model, X_train.columns


# 9. ADMISSION-TIME PREDICTION


admission_results, admission_model, admission_columns = (
    evaluate_scenario(
        admission_features,
        "Admission-Time Prediction"
    )
)

# 10. ADMISSION + FIRST-SEMESTER PREDICTION


first_sem_results, first_sem_model, first_sem_columns = (
    evaluate_scenario(
        first_sem_features,
        "Admission + First-Semester Prediction"
    )
)


# 11. PERFORMANCE COMPARISON


comparison_df = pd.DataFrame([
    admission_results,
    first_sem_results
])

print(comparison_df.round(4).to_string(index=False))


# 12. VISUALIZATION 1 — PERFORMANCE COMPARISON


print("\nGenerating Prediction Performance Comparison...")

plot_data = comparison_df.set_index("Scenario")[
    ["Accuracy", "Macro F1"]
]

fig, ax = plt.subplots(figsize=(10, 6))

plot_data.plot(
    kind="bar",
    width=0.6,
    ax=ax
)

ax.set_title(
    "Prediction Performance at Different Academic Stages"
)

ax.set_xlabel("Prediction Stage")
ax.set_ylabel("Performance Score")

ax.set_ylim(0, 1.1)
ax.tick_params(axis="x", rotation=0)

for p in ax.patches:

    height = p.get_height()

    if height > 0:

        ax.annotate(
            f"{height:.4f}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points"
        )

ax.legend(title="Evaluation Metric")

plt.tight_layout()


plt.show()


# 13. PERFORMANCE IMPROVEMENT


accuracy_improvement = (
    first_sem_results["Accuracy"] -
    admission_results["Accuracy"]
)

f1_improvement = (
    first_sem_results["Macro F1"] -
    admission_results["Macro F1"]
)



print(
    f"Accuracy Absolute Increase: {accuracy_improvement:.4f}"
)

print(
    f"Macro F1 Absolute Increase: {f1_improvement:.4f}"
)

print(
    f"Accuracy Improvement: {accuracy_improvement * 100:.2f} percentage points"
)

print(
    f"Macro F1 Improvement: {f1_improvement * 100:.2f} percentage points"
)


# 14. COEFFICIENT-BASED FEATURE IMPORTANCE


feature_importance = pd.DataFrame({
    "Feature": first_sem_columns,
    "Importance": np.mean(
        np.abs(first_sem_model.coef_),
        axis=0
    )
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)



top_15_features = feature_importance.head(15)

print(top_15_features.to_string(index=False))


# 15. VISUALIZATION 2 — TOP 15 FEATURE IMPORTANCE


print("\nGenerating Feature Importance Visualization...")

plot_features = top_15_features.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(11, 8))

bars = ax.barh(
    plot_features["Feature"],
    plot_features["Importance"]
)

ax.set_title(
    "Top 15 Features Influencing Student Outcome Predictions"
)

ax.set_xlabel(
    "Mean Absolute Logistic Regression Coefficient"
)

ax.set_ylabel("Features")

for bar in bars:

    width = bar.get_width()

    ax.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f" {width:.3f}",
        va="center"
    )

plt.tight_layout()



plt.show()


# 16. KEY FINDINGS


print("""
1. Admission-time prediction uses information available
   when students enter the institution.

2. Admission + first-semester prediction includes early
   academic performance information.

3. Both scenarios use the same train-test split and
   Logistic Regression algorithm.

4. Accuracy and Macro F1 are used to examine the
   difference between the two prediction stages.

5. Feature importance is calculated using the mean
   absolute Logistic Regression coefficients.

6. Higher coefficient magnitude indicates greater
   influence on the model's predictions, not causation.

7. Dummy categories belonging to the same original
   feature should not be interpreted as separate
   independent factors.

8. Prediction performance may improve when early
   academic information becomes available.
""")



# 17. FINAL SUMMARY


print("\nVisualizations Generated:")

print("1. Prediction Performance Comparison")
print("2. Top 15 Feature Importance")


print("\nBest Prediction Stage:")

if first_sem_results["Macro F1"] > admission_results["Macro F1"]:
    print("Admission + First-Semester Prediction")
else:
    print("Admission-Time Prediction")



