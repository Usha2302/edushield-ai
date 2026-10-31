
# OBJECTIVE 3 — MACHINE LEARNING MODEL DEVELOPMENT AND EVALUATION

# Research Question:
# How effectively can machine-learning algorithms predict student
# outcomes (Dropout, Enrolled and Graduate), and which model
# provides the best fit?

# Methods:
# - Logistic Regression
# - K-Nearest Neighbors
# - Decision Tree
# - Naive Bayes
# - Accuracy, Precision, Recall and F1-Score
# - Balanced Accuracy
# - Confusion Matrix
# - Classification Report
# - Model Performance Comparison


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sns.set_theme(style="whitegrid")
# 2. LOAD DATASET


df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)


# 3. FEATURE ENGINEERING

# First-semester academic features

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


# Financial features

df["Tuition_Overdue_Flag"] = (
    1 - df["Tuition fees up to date"]
)

df["Financial_Difficulty_Count"] = (
    df["Debtor"] + df["Tuition_Overdue_Flag"]
)

print("First-semester and financial features created.")


# 4. SELECT FEATURES AND TARGET

# Exclude second-semester information to prevent data leakage.

second_semester_columns = [
    column for column in df.columns
    if "2nd sem" in column.lower()
]

X = df.drop(
    columns=["Target"] + second_semester_columns
)

y = df["Target"]

print("Input Feature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

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
    "Tuition_Overdue_Flag"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))


# 6. ENCODE TARGET VARIABLE


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Target Class Mapping:")

for index, class_name in enumerate(label_encoder.classes_):
    print(class_name, ":", index)


# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# 8. ONE-HOT ENCODING

X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    dtype=int
)

# Align test columns with training columns

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("Training Features After Encoding:", X_train.shape[1])
print("Testing Features After Encoding:", X_test.shape[1])


# 9. FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler applied successfully.")
print("Training and testing data scaled.")


# 10. MACHINE LEARNING MODEL DEVELOPMENT

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Naive Bayes": GaussianNB()
}

# 11. MODEL TRAINING AND EVALUATION

results = []
confusion_matrices = {}

dropout_label = label_encoder.transform(["Dropout"])[0]

for model_name, model in models.items():

    print("\n" + "-" * 65)
    print("MODEL:", model_name)
    print("-" * 65)

    # Train model

    model.fit(X_train_scaled, y_train)

    # Prediction

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
        labels=[dropout_label],
        average="macro",
        zero_division=0
    )

    # Store results

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1-Score": macro_f1,
        "Dropout Recall": dropout_recall
    })

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_))
    )

    confusion_matrices[model_name] = cm

    print("\nAccuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1-Score:", round(macro_f1, 4))
    print("Dropout Recall:", round(dropout_recall, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )
# 12. FINAL MODEL EVALUATION


results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE SUMMARY:")
print(results_df.round(4).to_string(index=False))


# 13. BEST-FIT MODEL SELECTION


best_model_row = results_df.loc[
    results_df["Macro F1-Score"].idxmax()
]

best_model_name = best_model_row["Model"]

print("\nBEST-FIT MODEL:", best_model_name)

print(
    "Best Macro F1-Score:",
    round(best_model_row["Macro F1-Score"], 4)
)

print(
    "Accuracy:",
    round(best_model_row["Accuracy"], 4)
)

print(
    "Macro Precision:",
    round(best_model_row["Macro Precision"], 4)
)

print(
    "Macro Recall:",
    round(best_model_row["Macro Recall"], 4)
)

print(
    "Dropout Recall:",
    round(best_model_row["Dropout Recall"], 4)
)


# 14. VISUALIZATION 1: MODEL PERFORMANCE COMPARISON

performance_data = results_df.set_index("Model")[
    ["Accuracy", "Macro F1-Score"]
]

ax = performance_data.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

plt.title(
    "Machine Learning Model Performance Comparison",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)

plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")

plt.tight_layout()
plt.show()


# 15. VISUALIZATION 2: CONFUSION MATRIX COMPARISON

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 10)
)

axes = axes.flatten()

for index, (model_name, cm) in enumerate(
    confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=axes[index]
    )

    axes[index].set_title(
        model_name,
        fontweight="bold"
    )

    axes[index].set_xlabel("Predicted Outcome")
    axes[index].set_ylabel("Actual Outcome")


plt.suptitle(
    "Confusion Matrix Comparison of Machine Learning Models",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# 16. VISUALIZATION 3: DECISION TREE STRUCTURE

decision_tree_model = models["Decision Tree"]

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree_model,
    feature_names=X_train.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title(
    "Decision Tree Structure for Student Outcome Prediction",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# 17. FINAL SUMMARY

print("Dataset Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("Models Implemented:", len(models))

print("\nModels Used:")

for model_name in models:
    print("-", model_name)

print("\nBest-Fit Model:", best_model_name)

print("\nEvaluation Metrics:")
print("- Accuracy")
print("- Balanced Accuracy")
print("- Macro Precision")
print("- Macro Recall")
print("- Macro F1-Score")
print("- Dropout Recall")
print("- Confusion Matrix")
print("- Classification Report")

print("\nVisualizations Generated:")
print("1. Model Performance Comparison")
print("2. Confusion Matrix Comparison")
print("3. Decision Tree Structure")

print(
    "\nTarget Variable Excluded from Input Features:",
    "Target" not in X.columns
)

