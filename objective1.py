
# OBJECTIVE 1 — STATISTICAL ANALYSIS OF FACTORS ASSOCIATED
# WITH STUDENT ACADEMIC OUTCOMES

# Research Question:
# Which academic, demographic, admission-related and financial factors
# are significantly associated with student dropout, enrolment and graduation?

# Methods:
# - Descriptive statistical analysis
# - Kruskal-Wallis H-test
# - Chi-square test of independence
# - Effect-size analysis (Epsilon-squared and Cramer's V)
# - Benjamini-Hochberg correction for multiple hypothesis testing


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="whitegrid")


# 2. LOAD DATASET

df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)

target = "Target"
outcomes = ["Dropout", "Enrolled", "Graduate"]

print("\nSTUDENT OUTCOME DISTRIBUTION")
print("-" * 50)
print(df[target].value_counts().reindex(outcomes))

print("\nDESCRIPTIVE STATISTICS")
print("-" * 50)
print(df.describe().to_string())


# 3. DEFINE NUMERICAL VARIABLES

numerical_columns = [
    "Age at enrollment",
    "Previous qualification (grade)",
    "Admission grade",
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


# 4. DEFINE CATEGORICAL VARIABLES

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
    "International"
]


# 5. KRUSKAL-WALLIS TEST

print("\n" + "=" * 75)
print("1. KRUSKAL-WALLIS H-TEST")
print("=" * 75)

print("\nPurpose:")
print("Tests whether numerical-variable distributions differ")
print("across Dropout, Enrolled and Graduate groups.")

kruskal_results = []

for column in numerical_columns:

    groups = [
        df.loc[df[target] == outcome, column].dropna()
        for outcome in outcomes
    ]

    if all(len(group) > 0 for group in groups):

        statistic, p_value = kruskal(*groups)

        total_n = sum(len(group) for group in groups)
        number_groups = len(groups)

        epsilon_squared = (
            statistic - number_groups + 1
        ) / (total_n - number_groups)

        epsilon_squared = max(0, epsilon_squared)

        kruskal_results.append({
            "Variable": column,
            "H Statistic": statistic,
            "P Value": p_value,
            "Epsilon Squared": epsilon_squared
        })


kruskal_df = pd.DataFrame(kruskal_results)

if not kruskal_df.empty:

    kruskal_df["Adjusted P Value"] = multipletests(
        kruskal_df["P Value"],
        method="fdr_bh"
    )[1]

    kruskal_df["Significant"] = (
        kruskal_df["Adjusted P Value"] < 0.05
    )

    kruskal_df = kruskal_df.sort_values(
        "Adjusted P Value"
    )

    print("\nKRUSKAL-WALLIS TEST RESULTS")

    print(
        kruskal_df.to_string(
            index=False,
            float_format="%.6g"
        )
    )


# 6. CHI-SQUARE TEST OF INDEPENDENCE

print("\n" + "=" * 75)
print("2. CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 75)

print("\nPurpose:")
print("Tests whether each categorical variable is associated")
print("with student academic outcome.")

chi_square_results = []

for column in categorical_columns:

    contingency_table = pd.crosstab(
        df[column],
        df[target]
    )

    chi2, p_value, degrees_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    total_n = contingency_table.to_numpy().sum()

    rows, columns = contingency_table.shape
    minimum_dimension = min(rows - 1, columns - 1)

    if minimum_dimension > 0:

        cramers_v = np.sqrt(
            chi2 / (total_n * minimum_dimension)
        )

    else:
        cramers_v = np.nan

    minimum_expected = expected.min()

    percentage_below_five = (
        np.mean(expected < 5) * 100
    )

    # Expected-frequency conditions:
    # - No expected count below 1
    # - No more than 20% of expected counts below 5

    assumptions_met = (
        minimum_expected >= 1
        and percentage_below_five <= 20
    )

    chi_square_results.append({
        "Variable": column,
        "Chi-Square": chi2,
        "Degrees of Freedom": degrees_freedom,
        "P Value": p_value,
        "Cramers V": cramers_v,
        "Minimum Expected Count": minimum_expected,
        "Cells Below 5 (%)": percentage_below_five,
        "Assumptions Met": assumptions_met
    })


chi_df = pd.DataFrame(chi_square_results)

if not chi_df.empty:

    chi_df["Adjusted P Value"] = multipletests(
        chi_df["P Value"],
        method="fdr_bh"
    )[1]

    chi_df["Significant"] = (
        chi_df["Adjusted P Value"] < 0.05
    )

    chi_df["Reliable Significance"] = (
        chi_df["Significant"] &
        chi_df["Assumptions Met"]
    )

    chi_df = chi_df.sort_values(
        "Adjusted P Value"
    )

    print("\nCHI-SQUARE TEST RESULTS")

    print(
        chi_df.to_string(
            index=False,
            float_format="%.6g"
        )
    )


# 7. STATISTICAL SIGNIFICANCE SUMMARY

print("\n" + "=" * 75)
print("3. SUMMARY OF STATISTICAL FINDINGS")
print("=" * 75)

if not kruskal_df.empty:

    significant_numerical = kruskal_df[
        kruskal_df["Significant"]
    ]

    print("\nSignificant Numerical Variables:")

    print(
        len(significant_numerical),
        "out of",
        len(kruskal_df)
    )

    if not significant_numerical.empty:

        print(
            significant_numerical[
                [
                    "Variable",
                    "Adjusted P Value",
                    "Epsilon Squared"
                ]
            ].to_string(index=False)
        )

    else:
        print("No significant differences found.")


if not chi_df.empty:

    significant_categorical = chi_df[
        chi_df["Reliable Significance"]
    ]

    flagged_categorical = chi_df[
        ~chi_df["Assumptions Met"]
    ]

    print(
        "\nCategorical Variables with Reliable "
        "Significant Associations:"
    )

    print(
        len(significant_categorical),
        "out of",
        len(chi_df)
    )

    if not significant_categorical.empty:

        print(
            significant_categorical[
                [
                    "Variable",
                    "Adjusted P Value",
                    "Cramers V"
                ]
            ].to_string(index=False)
        )

    print("\nCategorical Variables Requiring Further Investigation:")

    if not flagged_categorical.empty:

        print(
            flagged_categorical[
                [
                    "Variable",
                    "Minimum Expected Count",
                    "Cells Below 5 (%)"
                ]
            ].to_string(index=False)
        )

    else:
        print("No categorical variables were flagged.")


# 8. VISUALIZATION 1 — LOLLIPOP CHART
# Strength of categorical associations

print("\n" + "=" * 75)
print("4. VISUALIZATION — CATEGORICAL ASSOCIATION STRENGTH")
print("=" * 75)

plot_data = chi_df[
    chi_df["Reliable Significance"]
].dropna(subset=["Cramers V"])

plot_data = (
    plot_data.nlargest(12, "Cramers V")
    .sort_values("Cramers V")
)

if not plot_data.empty:

    plt.figure(figsize=(11, 7))

    plt.hlines(
        y=plot_data["Variable"],
        xmin=0,
        xmax=plot_data["Cramers V"],
        linewidth=2
    )

    plt.scatter(
        plot_data["Cramers V"],
        plot_data["Variable"],
        s=90
    )

    for _, row in plot_data.iterrows():

        plt.text(
            row["Cramers V"] + 0.005,
            row["Variable"],
            f'{row["Cramers V"]:.3f}',
            va="center"
        )

    plt.title(
        "Categorical Factors Associated with Student Outcomes",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Cramer's V (Association Strength)")
    plt.ylabel("Categorical Factors")

    plt.tight_layout()
    plt.show()

else:
    print("No reliable significant categorical associations found.")


# 9. VISUALIZATION 2 — LINE CHART
# Semester-wise academic progression

print("\n" + "=" * 75)
print("5. VISUALIZATION — SEMESTER-WISE ACADEMIC PROGRESSION")
print("=" * 75)

semester_columns = [
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)"
]

semester_means = df.groupby(target)[
    semester_columns
].mean()

semester_means = semester_means.reindex(outcomes)

semester_means.index = [
    "Dropout",
    "Enrolled",
    "Graduate"
]

semester_means.columns = [
    "First Semester",
    "Second Semester"
]

print("\nMean Approved Curricular Units by Student Outcome:")
print(semester_means.to_string())

plt.figure(figsize=(10, 6))

for outcome in outcomes:

    plt.plot(
        semester_means.columns,
        semester_means.loc[outcome],
        marker="o",
        linewidth=2,
        markersize=8,
        label=outcome
    )

plt.title(
    "Mean Approved Curricular Units Across Student Outcomes",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Academic Semester")
plt.ylabel("Mean Number of Approved Curricular Units")

plt.legend(title="Student Outcome")
plt.tight_layout()
plt.show()


# 10. CHI-SQUARE ASSUMPTION SUMMARY

print("\n" + "=" * 75)
print("6. CHI-SQUARE EXPECTED-FREQUENCY CHECK")
print("=" * 75)

if not chi_df.empty:

    passed = chi_df[chi_df["Assumptions Met"]]
    flagged = chi_df[~chi_df["Assumptions Met"]]

    print("\nVariables passing expected-frequency conditions:")
    print(len(passed))

    print("\nVariables requiring further investigation:")
    print(len(flagged))

    if not flagged.empty:

        print(
            flagged[
                [
                    "Variable",
                    "Minimum Expected Count",
                    "Cells Below 5 (%)"
                ]
            ].to_string(index=False)
        )


# 11. INTERPRETATION NOTES



print("\n1. Adjusted p-values below 0.05 indicate statistical significance.")
print("2. Extremely small p-values displayed as 0 are not exactly zero;")
print("   they are below the displayed numerical precision.")
print("3. Effect sizes describe the strength of observed associations.")
print("4. Statistical significance does not establish causation.")
print("5. Kruskal-Wallis identifies group differences, not specific pairs.")
print("6. Chi-square results with sparse expected counts need caution.")
print("7. Economic indicators are excluded from individual-level")
print("   Kruskal-Wallis testing because values may repeat across students.")
print("8. Post-hoc testing is needed to identify which outcome groups")
print("   differ after a significant Kruskal-Wallis result.")
print("9. Semester-wise averages describe group-level patterns,")
print("   not individual student trajectories.")


# 12. FINAL SUMMARY

print("Numerical variables tested:", len(kruskal_results))
print("Categorical variables tested:", len(chi_square_results))

print("\nVisualizations generated:")
print("1. Lollipop Chart — Categorical Association Strength")
print("2. Line Chart — Semester-wise Academic Progression")

