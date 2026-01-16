# ======================================================
# Ravenstack Churn Analysis — Consolidated Pipeline
# ======================================================
#
# Research Question:
# Why do Ravenstack customers churn?
#
# Generated on: 2026-01-16T06:13:36.303088+00:00 UTC
#
# ======================================================


# ======================================================
# Dataset Consolidation
# ======================================================
# Structural unification of all Ravenstack datasets into a single analytical table.

import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
OUTPUT_PATH = "C:/Users/hp/Exploratory Data Analysis/Outputs/ravenstack_account_churn_consolidated.csv"
ANALYSIS_DATE = pd.Timestamp.today()

# -----------------------------
# Load datasets
# -----------------------------
accounts = pd.read_csv(
    "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_accounts.csv",
    parse_dates=["signup_date"]
)

# MECHANICAL FIX:
# Drop pre-existing churn_flag to avoid merge collision
accounts = accounts.drop(columns=["churn_flag"], errors="ignore")

churn_events = pd.read_csv(
    "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_churn_events.csv",
    parse_dates=["churn_date"]
)

subscriptions = pd.read_csv(
    "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_subscriptions.csv",
    parse_dates=["start_date", "end_date"]
)

usage = pd.read_csv(
    "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_feature_usage.csv",
    parse_dates=["usage_date"]
)

tickets = pd.read_csv(
    "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_support_tickets.csv",
    parse_dates=["submitted_at", "closed_at"]
)

# -----------------------------
# Step 1: Build churn label table
# -----------------------------
effective_churn = (
    churn_events[churn_events["is_reactivation"] == False]
    .sort_values("churn_date")
    .groupby("account_id", as_index=False)
    .last()
)

churn_labels = effective_churn[["account_id", "churn_date"]].copy()
churn_labels["churn_flag"] = 1

# -----------------------------
# Step 2: Build reference date per account
# -----------------------------
accounts_ref = accounts.merge(
    churn_labels,
    on="account_id",
    how="left"
)

accounts_ref["reference_date"] = accounts_ref["churn_date"].fillna(ANALYSIS_DATE)
accounts_ref["churn_flag"] = accounts_ref["churn_flag"].fillna(0).astype(int)

# -----------------------------
# Step 3: Subscription features (pre-reference)
# -----------------------------
subscriptions_ref = subscriptions.merge(
    accounts_ref[["account_id", "reference_date"]],
    on="account_id",
    how="left"
)

subscriptions_ref = subscriptions_ref[
    subscriptions_ref["start_date"] <= subscriptions_ref["reference_date"]
]

subscription_features = (
    subscriptions_ref
    .groupby("account_id")
    .agg(
        subscription_age_days=(
            "start_date",
            lambda x: (
                subscriptions_ref.loc[x.index, "reference_date"].iloc[0]
                - x.min()
            ).days
        ),
        avg_mrr=("mrr_amount", "mean"),
        has_upgraded=("upgrade_flag", "max"),
        has_downgraded=("downgrade_flag", "max"),
        active_subscription_count=("subscription_id", "nunique"),
        auto_renew_flag=("auto_renew_flag", "max")
    )
    .reset_index()
)

# -----------------------------
# Step 4: Usage features with rolling windows
# -----------------------------
usage = usage.merge(
    subscriptions[["subscription_id", "account_id"]],
    on="subscription_id",
    how="left"
)

usage = usage.merge(
    accounts_ref[["account_id", "reference_date"]],
    on="account_id",
    how="left"
)

usage_pre = usage[usage["usage_date"] <= usage["reference_date"]]

usage_features = (
    usage_pre
    .groupby("account_id")
    .apply(lambda df: pd.Series({
        "usage_events_30d": (
            df["usage_date"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=30)
        ).sum(),
        "usage_events_90d": (
            df["usage_date"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=90)
        ).sum(),
        "unique_features_30d": df[
            df["usage_date"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=30)
        ]["feature_name"].nunique(),
        "error_events_30d": df[
            df["usage_date"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=30)
        ]["error_count"].sum(),
        "days_since_last_usage": (
            df["reference_date"].iloc[0] - df["usage_date"].max()
        ).days
    }))
    .reset_index()
)

# -----------------------------
# Step 5: Support ticket features with windows
# -----------------------------
tickets = tickets.merge(
    accounts_ref[["account_id", "reference_date"]],
    on="account_id",
    how="left"
)

tickets_pre = tickets[tickets["submitted_at"] <= tickets["reference_date"]]

ticket_features = (
    tickets_pre
    .groupby("account_id")
    .apply(lambda df: pd.Series({
        "tickets_30d": (
            df["submitted_at"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=30)
        ).sum(),
        "tickets_90d": (
            df["submitted_at"]
            >= df["reference_date"].iloc[0] - pd.Timedelta(days=90)
        ).sum(),
        "avg_resolution_time": df["resolution_time_hours"].mean(),
        "escalation_rate": df["escalation_flag"].mean(),
        "open_ticket_flag": df["closed_at"].isna().any()
    }))
    .reset_index()
)

# -----------------------------
# Step 6: Assemble master table
# -----------------------------
master = (
    accounts_ref
    .drop(columns=["churn_date", "reference_date"])
    .merge(subscription_features, on="account_id", how="left")
    .merge(usage_features, on="account_id", how="left")
    .merge(ticket_features, on="account_id", how="left")
)

# -----------------------------
# Sanity checks
# -----------------------------
assert master["account_id"].is_unique, "Row explosion detected"
assert master.shape[0] == accounts.shape[0], "Account mismatch"

# -----------------------------
# Save output
# -----------------------------
master.to_csv(OUTPUT_PATH, index=False)



# ======================================================
# Data Integrity and Leakage Risk
# ======================================================
# Enforces causal discipline by detecting post-churn leakage and temporal violations.

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
entity_id = "account_id"
churn_col = "churn_flag"

paths = {
    "consolidated": "C:/Users/hp/Exploratory Data Analysis/Outputs/ravenstack_account_churn_consolidated.csv",
    "accounts": "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_accounts.csv",
    "churn_events": "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_churn_events.csv",
    "usage": "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_feature_usage.csv",
    "subscriptions": "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_subscriptions.csv",
    "tickets": "C:/Users/hp/Exploratory Data Analysis/CSV files/ravenstack_support_tickets.csv"
}

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(paths["consolidated"])
df_accounts = pd.read_csv(paths["accounts"])
df_churn = pd.read_csv(paths["churn_events"])
df_usage = pd.read_csv(paths["usage"])
df_subs = pd.read_csv(paths["subscriptions"])
df_tickets = pd.read_csv(paths["tickets"])

print("\n--- DATA LOADED ---")
print("Consolidated shape:", df.shape)

# =========================
# 1. DATASET IDENTITY & SHAPE DIAGNOSTICS
# =========================
print("\n--- DATASET IDENTITY DIAGNOSTICS ---")

identity_results = {
    "row_count": len(df),
    "column_count": df.shape[1],
    "unique_entities": df[entity_id].nunique(dropna=True),
    "entity_null_rate": df[entity_id].isna().mean(),
    "entity_duplication_factor": len(df) / df[entity_id].nunique(dropna=True),
    "full_row_duplicate_rate": df.duplicated().mean()
}

print(pd.Series(identity_results))

dup_counts = df[entity_id].value_counts()
print("\nEntities with duplicates:", (dup_counts > 1).sum())

# =========================
# 2. PRE-JOIN vs POST-JOIN ENTITY COVERAGE
# =========================
print("\n--- PRE vs POST JOIN COVERAGE ---")

def detect_entity_column(raw_df):
    candidates = [c for c in raw_df.columns if "account" in c.lower() and "id" in c.lower()]
    return candidates[0] if candidates else None

def entity_coverage(raw_df, name):
    detected_id = detect_entity_column(raw_df)

    if detected_id is None:
        return pd.Series({
            "rows": len(raw_df),
            "entity_column": None,
            "unique_entities": np.nan,
            "null_entity_rate": np.nan
        }, name=name)

    return pd.Series({
        "rows": len(raw_df),
        "entity_column": detected_id,
        "unique_entities": raw_df[detected_id].nunique(dropna=True),
        "null_entity_rate": raw_df[detected_id].isna().mean()
    }, name=name)

coverage = pd.concat([
    entity_coverage(df_accounts, "accounts"),
    entity_coverage(df_churn, "churn_events"),
    entity_coverage(df_usage, "usage"),
    entity_coverage(df_subs, "subscriptions"),
    entity_coverage(df_tickets, "tickets"),
    entity_coverage(df, "consolidated")
], axis=1)

print(coverage)

# =========================
# 3. TEMPORAL & LEAKAGE RED FLAG SCAN
# =========================
print("\n--- TEMPORAL / LEAKAGE RED FLAGS ---")

temporal_cols = [
    "signup_date",
    "days_since_last_usage",
    "subscription_age_days"
]

leakage_scan = {}
for col in temporal_cols:
    if col in df.columns:
        leakage_scan[col] = {
            "null_rate": df[col].isna().mean(),
            "min": df[col].min(),
            "max": df[col].max()
        }

print(pd.DataFrame(leakage_scan).T)

# =========================
# 4. POST-CHURN ACTIVITY CHECKS
# =========================
print("\n--- POST-CHURN ACTIVITY CHECKS ---")

post_churn_activity = df.loc[df[churn_col] == 1, [
    "usage_events_30d",
    "usage_events_90d",
    "tickets_30d",
    "tickets_90d",
    "open_ticket_flag"
]]

print(post_churn_activity.describe())

print("\nPost-churn non-zero rate:")
print((post_churn_activity > 0).mean())

# =========================
# 5. GLOBAL MISSINGNESS
# =========================
print("\n--- GLOBAL MISSINGNESS ---")

missingness = df.isna().mean().sort_values(ascending=False)
print(missingness[missingness > 0])

# =========================
# 6. MISSINGNESS BY CHURN (FUTURE-PROOF)
# =========================
print("\n--- MISSINGNESS BY CHURN ---")

missing_by_churn = (
    df
    .groupby(churn_col)
    .apply(lambda x: x.drop(columns=[churn_col]).isna().mean())
)


print(missing_by_churn.T)

# =========================
# 7. FEATURE DISTRIBUTION CHECKS
# =========================
print("\n--- FEATURE DISTRIBUTION CHECKS ---")

numeric_cols = df.select_dtypes(include="number").columns.drop(churn_col)

distribution_summary = (
    df
    .groupby(churn_col)[numeric_cols]
    .agg(["mean", "median"])
)

print(distribution_summary)

# =========================
# 8. EXPLICIT LEAKAGE CANDIDATES
# =========================
print("\n--- LEAKAGE CANDIDATES ---")

leakage_candidates = [
    "days_since_last_usage",
    "open_ticket_flag",
    "usage_events_30d",
    "usage_events_90d"
]

leakage_summary = (
    df
    .groupby(churn_col)[leakage_candidates]
    .mean()
)

print(leakage_summary)

print("\n--- DATA INTEGRITY & LEAKAGE DIAGNOSTICS COMPLETE ---")



# ======================================================
# Target Variable Analysis
# ======================================================
# Interrogates churn prevalence, timing, and cohort asymmetries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# Load data
# ----------------------------
file_path = "C:/Users/hp/Exploratory Data Analysis/Outputs/ravenstack_account_churn_consolidated.csv"
df = pd.read_csv(file_path)

df["signup_date"] = pd.to_datetime(df["signup_date"])

# ----------------------------
# Explicit churn definition
# churn = 1 means DID churn
# ----------------------------
df["churn"] = (df["churn_flag"] == 0).astype(int)

# ----------------------------
# A. CLASS BALANCE
# ----------------------------
overall_churn_rate = df["churn"].mean()
print(f"Overall churn rate: {overall_churn_rate:.4f}")

segment_cols = ["industry", "plan_tier", "is_trial", "country"]

for col in segment_cols:
    churn_by_segment = (
        df.groupby(col)["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()
    churn_by_segment.plot(kind="bar")
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# ----------------------------
# B. TEMPORAL CHURN BEHAVIOR
# ----------------------------

df["signup_month"] = df["signup_date"].dt.to_period("M").astype(str)

monthly_churn = df.groupby("signup_month")["churn"].mean()

plt.figure()
monthly_churn.plot()
plt.title("Churn Rate Over Time (by Signup Month)")
plt.ylabel("Churn Rate")
plt.xlabel("Signup Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Cohort-based churn curves
# ----------------------------

df["cohort"] = df["signup_date"].dt.to_period("M")
df["age_bucket"] = (df["subscription_age_days"] // 30) * 30

cohort_churn = (
    df.groupby(["cohort", "age_bucket"])["churn"]
    .mean()
    .reset_index()
)

for cohort, cohort_df in cohort_churn.groupby("cohort"):
    churn_curve = cohort_df.sort_values("age_bucket")["churn"].cumsum()
    survival_curve = 1 - churn_curve

    plt.plot(
        cohort_df.sort_values("age_bucket")["age_bucket"],
        survival_curve,
        label=str(cohort)
    )

plt.title("Cohort Survival Curves")
plt.xlabel("Customer Age (Days)")
plt.ylabel("Survival Probability")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ----------------------------
# Early-life vs Late-life churn
# ----------------------------

bins = [0, 30, 90, 180, np.inf]
labels = ["0–30", "31–90", "91–180", "180+"]

df["tenure_band"] = pd.cut(df["subscription_age_days"], bins=bins, labels=labels)

tenure_churn = df.groupby("tenure_band")["churn"].mean()

plt.figure()
tenure_churn.plot(kind="bar")
plt.title("Churn Rate by Tenure Band")
plt.ylabel("Churn Rate")
plt.xlabel("Tenure Band")
plt.tight_layout()
plt.show()

# ----------------------------
# C. BASELINE PERFORMANCE
# ----------------------------

y_true = df["churn"]

# Naive baseline: predict NO churn
y_pred_naive = np.zeros(len(df))

print("Naive Baseline Performance (Predict No Churn)")
print("Accuracy:", accuracy_score(y_true, y_pred_naive))
print("Precision:", precision_score(y_true, y_pred_naive, zero_division=0))
print("Recall:", recall_score(y_true, y_pred_naive))
print("F1:", f1_score(y_true, y_pred_naive, zero_division=0))

# ----------------------------
# Heuristic baseline
# ----------------------------

heuristic_pred = (
    (df["days_since_last_usage"] > 30) |
    (df["usage_events_30d"] == 0)
).astype(int)

print("\nHeuristic Baseline Performance")
print("Accuracy:", accuracy_score(y_true, heuristic_pred))
print("Precision:", precision_score(y_true, heuristic_pred))
print("Recall:", recall_score(y_true, heuristic_pred))
print("F1:", f1_score(y_true, heuristic_pred))



# ======================================================
# Feature Triage and Temporal Admissibility
# ======================================================
# Filters features based on temporal availability and behavioral interpretability.

import pandas as pd

# ------------------------------------------------------------
# Step 4.1 — Feature Triage & Temporal Admissibility
# Diagnostic-only. NO dataset mutations.
# ------------------------------------------------------------

OUTPUT_PATH = (
    r"C:\Users\hp\Exploratory Data Analysis\Outputs"
    r"\feature_triage_temporal_admissibility_registry.csv"
)

# ------------------------------------------------------------
# Feature Registry Construction
# ------------------------------------------------------------

feature_registry = []

def register_feature(
    feature,
    temporal_admissibility,
    exploratory_priority,
    rationale,
    notes=""
):
    feature_registry.append({
        "feature": feature,
        "temporal_admissibility": temporal_admissibility,
        "exploratory_priority": exploratory_priority,
        "rationale": rationale,
        "notes": notes
    })


# ------------------------
# Identifiers (Excluded)
# ------------------------
register_feature(
    "account_id",
    "Pre-decision admissible",
    "Discard",
    "Pure identifier with no behavioral or predictive content."
)

register_feature(
    "account_name",
    "Pre-decision admissible",
    "Discard",
    "Free-text identifier; unstable and non-interpretable."
)

# ------------------------
# Static Context Features
# ------------------------
register_feature(
    "industry",
    "Pre-decision admissible",
    "Secondary",
    "Weak-to-moderate contextual signal; does not explain churn timing.",
    "Use only as conditioning variable."
)

register_feature(
    "country",
    "Pre-decision admissible",
    "Secondary",
    "Strong segmentation observed but likely confounded.",
    "May proxy infrastructure or pricing effects."
)

register_feature(
    "referral_source",
    "Pre-decision admissible",
    "Secondary",
    "Acquisition context; limited behavioral relevance."
)

register_feature(
    "plan_tier",
    "Pre-decision admissible",
    "Secondary",
    "Contextual plan information; insufficient alone."
)

register_feature(
    "seats",
    "Pre-decision admissible",
    "Secondary",
    "Account size proxy with weak causal linkage."
)

register_feature(
    "is_trial",
    "Pre-decision admissible",
    "Discard",
    "Empirically negligible separation in target analysis."
)

# ------------------------
# Lifecycle / Tenure
# ------------------------
register_feature(
    "signup_date",
    "Ambiguous",
    "Discard",
    "Implicit lifecycle encoding; unsafe without horizon control.",
    "Tenure analysis previously invalid."
)

register_feature(
    "subscription_age_days",
    "Ambiguous",
    "Discard",
    "Explicit lifecycle feature with binning artifacts.",
    "Excluded per failed tenure analysis."
)

# ------------------------
# Revenue & Contract State
# ------------------------
register_feature(
    "avg_mrr",
    "Pre-decision admissible",
    "Secondary",
    "Economic context; not behaviorally decisive."
)

register_feature(
    "has_upgraded",
    "Outcome-relative",
    "Discard",
    "Likely downstream of churn risk or recovery attempts."
)

register_feature(
    "has_downgraded",
    "Outcome-relative",
    "Discard",
    "Often precedes or follows churn decision."
)

register_feature(
    "auto_renew_flag",
    "Pre-decision admissible",
    "Secondary",
    "Contractual setting; contextual only."
)

register_feature(
    "active_subscription_count",
    "Ambiguous",
    "Discard",
    "May encode churn state implicitly."
)

# ------------------------
# Behavioral Usage (Core)
# ------------------------
register_feature(
    "usage_events_30d",
    "Pre-decision admissible",
    "High-priority",
    "Direct engagement signal; candidate for incremental value beyond inactivity."
)

register_feature(
    "usage_events_90d",
    "Pre-decision admissible",
    "High-priority",
    "Captures longer-term engagement decay."
)

register_feature(
    "unique_features_30d",
    "Pre-decision admissible",
    "High-priority",
    "Usage depth and breadth; may precede inactivity collapse."
)

register_feature(
    "days_since_last_usage",
    "Outcome-relative",
    "Discard",
    "Core inactivity heuristic already dominates prediction.",
    "Direct use risks circularity."
)

# ------------------------
# Friction & Support
# ------------------------
register_feature(
    "error_events_30d",
    "Pre-decision admissible",
    "High-priority",
    "Product friction may precede disengagement."
)

register_feature(
    "tickets_30d",
    "Pre-decision admissible",
    "Secondary",
    "Noisy dissatisfaction proxy."
)

register_feature(
    "tickets_90d",
    "Pre-decision admissible",
    "Secondary",
    "Weaker temporal alignment."
)

register_feature(
    "avg_resolution_time",
    "Ambiguous",
    "Discard",
    "Defined post-ticket lifecycle."
)

register_feature(
    "escalation_rate",
    "Ambiguous",
    "Discard",
    "Downstream resolution artifact."
)

register_feature(
    "open_ticket_flag",
    "Outcome-relative",
    "Discard",
    "May exist after churn-triggering events."
)

# ------------------------
# Target Variable
# ------------------------
register_feature(
    "churn_flag",
    "Outcome",
    "Excluded",
    "Target variable; never eligible for feature analysis."
)

# ------------------------------------------------------------
# Final Registry Artifact
# ------------------------------------------------------------

feature_registry_df = (
    pd.DataFrame(feature_registry)
      .sort_values(
          by=["temporal_admissibility", "exploratory_priority", "feature"]
      )
      .reset_index(drop=True)
)

# ------------------------------------------------------------
# Guaranteed Output
# ------------------------------------------------------------

# 1. Display
print("\n=== FEATURE TRIAGE & TEMPORAL ADMISSIBILITY REGISTRY ===\n")
print(feature_registry_df.to_string(index=False))

# 2. Persist
feature_registry_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nRegistry saved to:\n{OUTPUT_PATH}")




# ======================================================
# Layer 2 — Conditional Signal Variation
# ======================================================
# Examines how churn signals change under pricing, renewal, and geographic conditions.

import pandas as pd

# -------------------------------
# Layer 1: Setup & Invariants
# -------------------------------

DATA_PATH = r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated (Updated).csv"
df = pd.read_csv(DATA_PATH)

TARGET = "churn_flag"

PRIMARY_FEATURES = [
    "usage_events_30d",
    "usage_events_90d",
    "unique_features_30d",
    "error_events_30d"
]

CONDITIONAL_FEATURES = [
    "auto_renew_flag",
    "avg_mrr",
    "country",
    "industry",
    "plan_tier",
    "referral_source",
    "seats",
    "tickets_30d",
    "tickets_90d"
]

# Features forbidden from entering the analytical frame at all
FORBIDDEN_FRAME_FEATURES = [
    "days_since_last_usage",
    "subscription_age_days",
    "signup_date",
    "has_upgraded",
    "has_downgraded",
    "open_ticket_flag",
    "avg_resolution_time",
    "escalation_rate",
    "active_subscription_count",
    "account_id",
    "account_name"
]

ALLOWED_FEATURES = PRIMARY_FEATURES + CONDITIONAL_FEATURES
analysis_columns = ALLOWED_FEATURES + [TARGET]

df_analysis = df[analysis_columns].copy()

# -------------------------------
# Guardrails
# -------------------------------

assert TARGET in df_analysis.columns, "Target variable missing."

for col in FORBIDDEN_FRAME_FEATURES:
    assert col not in df_analysis.columns, f"Forbidden feature leaked in: {col}"

unexpected_cols = set(df_analysis.columns) - set(analysis_columns)
assert len(unexpected_cols) == 0, f"Unexpected columns found: {unexpected_cols}"

print("Layer 1 complete: Analytical frame locked.")
print(f"Primary features: {PRIMARY_FEATURES}")
print(f"Conditional features: {CONDITIONAL_FEATURES}")
print("Target present and protected. No forbidden features leaked.")
# =========================
import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
path = r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated.csv"
df = pd.read_csv(path)

# -------------------------------
# Sanity check: churn_flag exists and is binary
# -------------------------------
assert "churn_flag" in df.columns, "churn_flag column not found"
assert set(df["churn_flag"].dropna().unique()).issubset({0, 1}), \
    "churn_flag contains non-binary values"

# -------------------------------
# Normalize target
# Original:
#   churn_flag = 0 → churned
#   churn_flag = 1 → retained
#
# New:
#   churn = 1 → churned
#   churn = 0 → retained
# -------------------------------
df["churn"] = 1 - df["churn_flag"]

# -------------------------------
# Post-condition checks
# -------------------------------
assert set(df["churn"].unique()).issubset({0, 1}), "Normalized churn is not binary"

# Optional: distribution check (prints to console)
print("Original churn_flag distribution:")
print(df["churn_flag"].value_counts(normalize=True).rename("proportion"))
print("\nNormalized churn distribution:")
print(df["churn"].value_counts(normalize=True).rename("proportion"))

# -------------------------------
# Save updated dataset
# -------------------------------
updated_path = r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated (Updated).csv"
df.to_csv(updated_path, index=False)

print(f"\nUpdated dataset saved to:\n{updated_path}")

# =========================
# Layer 2: Conditional Signal Validation
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -------------------------
# Invariants
# -------------------------
TARGET = "churn"  # 1 = churned, 0 = retained

primary_features = [
    "usage_events_30d",
    "usage_events_90d",
    "unique_features_30d",
    "error_events_30d"
]

conditional_features = [
    "auto_renew_flag",
    "avg_mrr",
    "country",
    "industry",
    "plan_tier",
    "referral_source",
    "seats",
    "tickets_30d",
    "tickets_90d"
]

DATA_PATH = r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated (Updated).csv"
df = pd.read_csv(DATA_PATH)

# -------------------------
# Phase 2A — Visual diagnostics (bivariate only)
# -------------------------

for feature in conditional_features:
    if df[feature].dtype in ["int64", "float64"]:
        # Numeric → quantile bins
        df["_bin"] = pd.qcut(df[feature], q=10, duplicates="drop")

        churn_rate = (
            df.groupby("_bin")[TARGET]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(8, 4))
        plt.plot(
            churn_rate["_bin"].astype(str),
            churn_rate[TARGET],
            marker="o"
        )
        plt.xticks(rotation=45)
        plt.title(f"Churn Rate by {feature} (Quantile Bins)")
        plt.ylabel("Churn Rate")
        plt.xlabel(feature)
        plt.tight_layout()
        plt.show()

        df.drop(columns="_bin", inplace=True)

    else:
        # Categorical
        churn_rate = (
            df.groupby(feature)[TARGET]
            .mean()
            .sort_values(ascending=False)
        )

        churn_rate.plot(
            kind="bar",
            figsize=(8, 4),
            title=f"Churn Rate by {feature}"
        )
        plt.ylabel("Churn Rate")
        plt.tight_layout()
        plt.show()

# -------------------------
# Phase 2B — Numeric signal summary (Spearman)
# -------------------------

numeric_conditionals = [
    f for f in conditional_features
    if df[f].dtype in ["int64", "float64"]
]

lift_records = []

for feature in numeric_conditionals:
    corr, p = spearmanr(df[feature], df[TARGET])
    lift_records.append({
        "feature": feature,
        "type": "numeric",
        "spearman_corr": corr,
        "p_value": p
    })

lift_df = (
    pd.DataFrame(lift_records)
    .sort_values("spearman_corr", ascending=False)
    .round(4)
    .reset_index(drop=True)
)

print("\n=== Numeric Conditional Signal Summary (Spearman) ===")
print(lift_df)

# -------------------------
# Phase 2C — Redundancy check vs inactivity heuristic
# -------------------------

inactivity_proxy = "usage_events_30d"

redundancy_records = []

for feature in numeric_conditionals:
    corr, _ = spearmanr(df[feature], df[inactivity_proxy])
    redundancy_records.append({
        "feature": feature,
        "spearman_corr_with_inactivity": corr
    })

redundancy_df = (
    pd.DataFrame(redundancy_records)
    .round(4)
    .sort_values("spearman_corr_with_inactivity", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Redundancy vs Inactivity Heuristic ===")
print(redundancy_df)

# -------------------------
# Phase 2D — Controlled explanatory persistence
# (Explanatory only — not evaluative)
# -------------------------

coef_records = []

for cond in numeric_conditionals:
    cols = primary_features + [cond]
    X = df[cols]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000
    )
    model.fit(X_scaled, y)

    coef = model.coef_[0][-1]

    coef_records.append({
        "feature": cond,
        "coef": coef,
        "abs_coef": abs(coef)
    })

coef_df_clean = (
    pd.DataFrame(coef_records)
    .round(4)
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Conditional Feature Persistence (Controlled, Explanatory) ===")
print(coef_df_clean)



# ======================================================
# Layer 3 — Conditional Signal Validation
# ======================================================
# Stress-tests conditional signals for robustness and spurious correlation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated (Updated).csv"
TARGET = "churn_flag"  # unchanged definition

PRIMARY_FEATURES = {
    "usage_events_30d": 5,
    "usage_events_90d": 5,
    "unique_features_30d": 5,
    "error_events_30d": 4
}

# Conditional features mapped to tests
CONDITIONAL_TESTS = {
    "tickets_30d": {
        "primary": "usage_events_30d",
        "bins": 5
    },
    "tickets_90d": {
        "primary": "usage_events_30d",
        "bins": 5
    },
    "avg_mrr": {
        "primary": "usage_events_90d",
        "bins": 5
    },
    "auto_renew_flag": {
        "primary": "usage_events_30d",
        "filter_true_only": True
    },
    "seats": {
        "primary": "usage_events_30d",
        "bins": 5
    },
    "industry": {
        "primary": "usage_events_90d"
    },
    "country": {
        "primary": "usage_events_90d"
    }
}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Safety check
assert TARGET in df.columns, "Target missing"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def bin_feature(series, n_bins):
    return pd.qcut(series, q=n_bins, duplicates="drop")

def churn_table(df, group_cols):
    return (
        df.groupby(group_cols, observed=True)[TARGET]
          .mean()
          .reset_index()
          .rename(columns={TARGET: "churn_rate"})
    )

def plot_churn(df, x_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    for key, g in df.groupby(hue_col):
        plt.plot(g[x_col].astype(str), g["churn_rate"], marker="o", label=str(key))
    plt.title(title)
    plt.ylabel("Churn Rate")
    plt.xlabel(x_col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------
# LAYER 3 CORE LOOP
# -----------------------------
layer3_outputs = {}

for cond_feature, spec in CONDITIONAL_TESTS.items():
    primary = spec["primary"]
    df_work = df.copy()

    # Optional filter (auto-renew asymmetry test)
    if spec.get("filter_true_only", False):
        df_work = df_work[df_work["auto_renew_flag"] == True]

    # Bin primary behavior (freeze behavior)
    primary_bins = PRIMARY_FEATURES.get(primary, 5)
    df_work["_primary_bin"] = bin_feature(df_work[primary], primary_bins)

    # Bin conditional if numeric
    if pd.api.types.is_numeric_dtype(df_work[cond_feature]):
        bins = spec.get("bins", 5)
        df_work["_cond_bin"] = bin_feature(df_work[cond_feature], bins)
        cond_col = "_cond_bin"
    else:
        cond_col = cond_feature

    # Compute churn table
    churn_tbl = churn_table(
        df_work.dropna(subset=["_primary_bin", cond_col]),
        ["_primary_bin", cond_col]
    )

    layer3_outputs[cond_feature] = churn_tbl

    # Plot diagnostic contrast
    plot_churn(
        churn_tbl,
        x_col="_primary_bin",
        hue_col=cond_col,
        title=f"Layer 3 — {cond_feature} conditioned on {primary}"
    )

    # Print numeric diagnostic
    print(f"\n=== Layer 3 Diagnostic Table: {cond_feature} | Primary: {primary} ===")
    print(churn_tbl.sort_values(["_primary_bin", "churn_rate"]))

# -----------------------------
# END OF LAYER 3
# -----------------------------



# ======================================================
# Layer 4 — Behavioral State Logic
# ======================================================
# Synthesizes findings into behavioral trajectories that precede churn.

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(
    r"C:\Users\hp\Exploratory Data Analysis\Outputs\ravenstack_account_churn_consolidated (Updated).csv",
    dayfirst=True
)

# -------------------------------
# Layer 4 — Behavioral State Logic
# -------------------------------

# Behavioral states from usage (validated in Layer 3)
def behavior_state(row):
    if row["usage_events_30d"] <= 1:
        return "Dormant"
    elif row["usage_events_30d"] <= 3:
        return "Hard Disengagement"
    elif row["usage_events_30d"] <= 6:
        return "Soft Disengagement"
    else:
        return "Active"

df["behavior_state"] = df.apply(behavior_state, axis=1)

# -------------------------------
# Layer 4 — Mechanism Construction
# -------------------------------

df["behavioral_exhaustion_flag"] = (
    (df["behavior_state"].isin(["Hard Disengagement", "Dormant"])) &
    (df["usage_events_90d"] <= df["usage_events_30d"])
).astype(int)

df["economic_misalignment_flag"] = (
    (df["avg_mrr"] < df["avg_mrr"].median()) &
    (df["behavior_state"].isin(["Soft Disengagement", "Hard Disengagement"]))
).astype(int)

df["support_distress_flag"] = (
    (df["tickets_30d"] > 0) &
    (df["avg_resolution_time"] > df["avg_resolution_time"].median())
).astype(int)

df["structural_risk_flag"] = (
    (df["auto_renew_flag"] == True) &
    (df["seats"] < df["seats"].median())
).astype(int)

df["mechanism_score"] = (
    df["behavioral_exhaustion_flag"] +
    df["economic_misalignment_flag"] +
    df["support_distress_flag"] +
    df["structural_risk_flag"]
)

# -------------------------------
# Layer 4 — Figures
# -------------------------------

print("Total accounts:", len(df))
print("Overall churn rate:", df["churn"].mean())

mechanism_cols = [
    "behavioral_exhaustion_flag",
    "economic_misalignment_flag",
    "support_distress_flag",
    "structural_risk_flag",
    "mechanism_score"
]

print("\nMechanism prevalence:")
print(df[mechanism_cols].mean().sort_values(ascending=False))

for col in mechanism_cols[:-1]:
    print(f"\nChurn rate by {col}:")
    print(df.groupby(col)["churn"].mean())

print("\nChurn by mechanism score:")
print(
    df.groupby("mechanism_score")["churn"]
      .agg(["count", "mean"])
      .rename(columns={"mean": "churn_rate"})
)

print("\nMechanism score distribution:")
print(df["mechanism_score"].value_counts().sort_index())



# ======================================================
# Behavioral State Machine (Explicit)
# ======================================================
# Behavioral State Machine
# ------------------------
# 
# State 0: Healthy Engagement
# - High feature usage
# - Low error and support interaction
# - Stable renewal behavior
# 
# State 1: Friction Accumulation
# - Declining usage velocity
# - Rising error events or support tickets
# - Early disengagement signals
# 
# State 2: Behavioral Withdrawal
# - Sharp reduction in product interaction
# - Feature abandonment
# - Weak response to recovery signals
# 
# State 3: Churn
# - Subscription termination
# - Account inactivity
# - Terminal absorbing state
# 
# Transitions are monotonic and directional.
# Recovery is rare once State 2 is entered.

