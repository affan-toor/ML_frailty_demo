"""
build_frailty_index_nhanes.py

Builds, for adults aged 60+ in NHANES 2013-2014:
  Step 1-2: a Rockwood-style deficit-accumulation frailty index from
            questionnaire/exam/lab data (register-style data).
  Step 3:   personal wearable-activity features from wrist-accelerometer
            data for the same participants (daily-life activity data).
  Step 4:   a combined model linking activity patterns to frailty status,
            with basic performance and calibration checks.

Register/questionnaire data source (official NHANES):
  https://wwwn.cdc.gov/nchs/nhanes/   (2013-2014 cycle, "_H" files)

Wearable activity data source (pre-processed from the official NHANES
PAXMIN_H minute-level wrist-accelerometer release):
  "Minute level step counts and physical activity data from NHANES
  2011-2014", PhysioNet v1.0.2 (see README.md for the full citation,
  copied from the project's own citation panel).
  https://physionet.org/content/minute-level-step-count-nhanes/1.0.2/

IMPORTANT
- Variable names/codes follow the published NHANES 2013-2014 codebooks.
  Before using results beyond a personal demo, check each variable against
  its codebook page at https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/<FILE>.htm
- The accelerometer files are large (~250 MB compressed each). The script
  downloads them once, caches them locally, and reads them in chunks,
  keeping only the ~60+ age cohort, to avoid loading the full file into
  memory.
- First run will take a few minutes depending on your connection.

Tested with: Python 3.10+, pandas, numpy, requests, matplotlib, scikit-learn
(all included in a standard Anaconda install). No extra packages needed.

Run in Spyder: open this file and press F5.
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("NOTE: 'shap' is not installed. Step 5 (explainability) will be "
          "skipped. Install it with:  pip install shap   (or, in an Anaconda "
          "prompt: conda install -c conda-forge shap) and re-run.")

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.getcwd(), "nhanes_data")
os.makedirs(DATA_DIR, exist_ok=True)

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/"
REGISTER_FILES = ["DEMO_H", "MCQ_H", "DIQ_H", "BPQ_H", "HUQ_H", "DPQ_H", "BMX_H"]

PHYSIONET_BASE = "https://physionet.org/files/minute-level-step-count-nhanes/1.0.2/csv/"
ACC_VARS = ["PAXMTSM", "PAXPREDM"]  # activity intensity (MIMS), wear/sleep/nonwear flag

MIN_AGE = 60          # restrict to older adults, matching frailty literature
MIN_COMPLETE = 0.80   # require >=80% of deficit items answered
SEDENTARY_MIMS_THRESHOLD = 10.0   # low-intensity cutoff during wake wear
MIN_WAKE_MINUTES_PER_DAY = 600    # >=10h wake wear to count as a valid day
MIN_VALID_DAYS = 3                # need >=3 valid days for a personal baseline


# ---------------------------------------------------------------------------
# STEP 1: Download (with local caching) and load each register/questionnaire
#         NHANES file
# ---------------------------------------------------------------------------
def download_and_load_xpt(fname):
    local_path = os.path.join(DATA_DIR, fname + ".XPT")
    if not os.path.exists(local_path):
        url = NHANES_BASE + fname + ".XPT"
        print(f"Downloading {fname} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return pd.read_sas(local_path, format="xport")


print("Loading NHANES 2013-2014 register/questionnaire files...")
data = {f: download_and_load_xpt(f) for f in REGISTER_FILES}

df = data["DEMO_H"][["SEQN", "RIDAGEYR", "RIAGENDR"]]
for f in REGISTER_FILES[1:]:
    df = df.merge(data[f], on="SEQN", how="left")

df = df[df["RIDAGEYR"] >= MIN_AGE].reset_index(drop=True)
print(f"Participants aged {MIN_AGE}+: {len(df)}")


# ---------------------------------------------------------------------------
# STEP 2: Recode items into 0-1 deficits and compute the frailty index
# ---------------------------------------------------------------------------
def binary_deficit(series):
    return series.map({1: 1.0, 2: 0.0}).where(series.isin([1, 2]))

def graded_deficit_0to3(series):
    valid = series.isin([0, 1, 2, 3])
    return (series / 3.0).where(valid)

deficits = pd.DataFrame({"SEQN": df["SEQN"]})

mcq_items = {
    "MCQ160A": "arthritis", "MCQ160B": "chf", "MCQ160C": "chd",
    "MCQ160D": "angina", "MCQ160E": "heart_attack", "MCQ160F": "stroke",
    "MCQ160G": "emphysema", "MCQ160K": "chronic_bronchitis",
    "MCQ160L": "liver_condition", "MCQ160M": "thyroid_problem",
    "MCQ220": "cancer",
}
for col, name in mcq_items.items():
    if col in df.columns:
        deficits[name] = binary_deficit(df[col])

if "DIQ010" in df.columns:
    diq = df["DIQ010"]
    deficits["diabetes"] = diq.map({1: 1.0, 2: 0.0, 3: 0.5}).where(diq.isin([1, 2, 3]))

if "BPQ020" in df.columns:
    deficits["hypertension"] = binary_deficit(df["BPQ020"])
if "BPQ080" in df.columns:
    deficits["high_cholesterol"] = binary_deficit(df["BPQ080"])

if "HUQ010" in df.columns:
    huq = df["HUQ010"]
    deficits["fair_or_poor_health"] = huq.map(
        {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}
    ).where(huq.isin([1, 2, 3, 4, 5]))

dpq_cols = [c for c in df.columns if c.startswith("DPQ0") and c.endswith("0")]
for col in dpq_cols:
    deficits[col.lower()] = graded_deficit_0to3(df[col])

if "BMXBMI" in df.columns:
    deficits["obesity"] = (df["BMXBMI"] >= 30).astype(float)
    deficits.loc[df["BMXBMI"].isna(), "obesity"] = pd.NA

deficit_cols = [c for c in deficits.columns if c != "SEQN"]
print(f"Number of deficit items used: {len(deficit_cols)}")

n_measured = deficits[deficit_cols].notna().sum(axis=1)
n_present = deficits[deficit_cols].sum(axis=1, skipna=True)
frailty_index = n_present / n_measured
enough_data = (n_measured / len(deficit_cols)) >= MIN_COMPLETE

result = pd.DataFrame({
    "SEQN": deficits["SEQN"].astype(int),
    "age": df["RIDAGEYR"],
    "sex": df["RIAGENDR"].map({1: "male", 2: "female"}),
    "n_deficits_measured": n_measured,
    "frailty_index": frailty_index,
})
result = result[enough_data].reset_index(drop=True)

result["frailty_category"] = pd.cut(
    result["frailty_index"],
    bins=[-0.01, 0.10, 0.21, 1.01],
    labels=["robust", "pre-frail", "frail"],
)

print("\nFrailty index summary:")
print(result["frailty_index"].describe())
print("\nFrailty category counts:")
print(result["frailty_category"].value_counts())

fi_path = os.path.join(os.getcwd(), "nhanes_frailty_index.csv")
result.to_csv(fi_path, index=False)
print(f"Saved: {fi_path}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(result["frailty_index"].dropna(), bins=30, color="steelblue")
axes[0].set_title("Distribution of frailty index")
axes[0].set_xlabel("Frailty index")
axes[0].set_ylabel("Count")
result.boxplot(column="frailty_index", by="sex", ax=axes[1])
axes[1].set_title("Frailty index by sex")
axes[1].set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# STEP 3: Wearable activity data (wrist accelerometer, 2013-2014 cycle)
#         Downloaded in chunks and filtered to our 60+ cohort only, to keep
#         memory use reasonable (the full files cover the whole sample).
# ---------------------------------------------------------------------------
min_cols = [f"min_{i:04d}" for i in range(1, 1441)]
usecols = ["SEQN", "PAXDAYM", "PAXDAYWM"] + min_cols
keep_ids = set(result["SEQN"])

def download_physionet_csv(varname):
    local_path = os.path.join(DATA_DIR, f"nhanes_1440_{varname}.csv.xz")
    if not os.path.exists(local_path):
        url = PHYSIONET_BASE + f"nhanes_1440_{varname}.csv.xz"
        print(f"Downloading {varname} (~250MB compressed, may take a few minutes)...")
        try:
            r = requests.get(url, timeout=600, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Could not auto-download {varname}. Please download it manually "
                f"from {url} and save it to {local_path}, then re-run."
            ) from e
    return local_path

def load_and_filter(varname, keep_ids, chunksize=5000):
    path = download_physionet_csv(varname)
    print(f"Reading {varname} in chunks and keeping only our {len(keep_ids)} "
          f"participants (this avoids loading the full file into memory)...")
    kept = []
    for chunk in pd.read_csv(path, compression="xz", usecols=usecols, chunksize=chunksize):
        chunk["SEQN"] = chunk["SEQN"].astype(int)
        filtered = chunk[chunk["SEQN"].isin(keep_ids)]
        if len(filtered):
            kept.append(filtered)
    return pd.concat(kept, ignore_index=True)

mims = load_and_filter("PAXMTSM", keep_ids)    # per-minute activity intensity (MIMS)
wear = load_and_filter("PAXPREDM", keep_ids)   # per-minute wear/sleep/nonwear code

# align rows (both files share the same SEQN/PAXDAYM structure)
mims = mims.sort_values(["SEQN", "PAXDAYM"]).reset_index(drop=True)
wear = wear.sort_values(["SEQN", "PAXDAYM"]).reset_index(drop=True)
assert (mims["SEQN"].values == wear["SEQN"].values).all()
assert (mims["PAXDAYM"].values == wear["PAXDAYM"].values).all()

wear_vals = wear[min_cols].to_numpy()
mims_vals = mims[min_cols].to_numpy()

is_wake_wear = (wear_vals == 1)  # 1 = wake wear, 2 = sleep wear, 3 = unknown, 4 = nonwear
total_wake_minutes = is_wake_wear.sum(axis=1)
sedentary_minutes = ((wear_vals == 1) & (mims_vals < SEDENTARY_MIMS_THRESHOLD)).sum(axis=1)
active_minutes = total_wake_minutes - sedentary_minutes
daily_activity_sum = np.where(is_wake_wear, mims_vals, 0).sum(axis=1)

daily = pd.DataFrame({
    "SEQN": mims["SEQN"],
    "PAXDAYM": mims["PAXDAYM"],
    "total_wake_minutes": total_wake_minutes,
    "active_minutes": active_minutes,
    "sedentary_minutes": sedentary_minutes,
    "daily_activity_sum": daily_activity_sum,
})
daily = daily[daily["total_wake_minutes"] >= MIN_WAKE_MINUTES_PER_DAY]

# ---- personal baseline across each participant's own days ----
# Mirrors a contextual/personal-baseline approach: summarise each person's
# own pattern first, then look at how variable it is from day to day,
# rather than comparing everyone to one population-wide threshold.
person_activity = daily.groupby("SEQN").agg(
    n_valid_days=("PAXDAYM", "count"),
    mean_daily_activity=("daily_activity_sum", "mean"),
    sd_daily_activity=("daily_activity_sum", "std"),
    mean_active_minutes=("active_minutes", "mean"),
    mean_sedentary_minutes=("sedentary_minutes", "mean"),
).reset_index()

person_activity["activity_cv"] = (
    person_activity["sd_daily_activity"] / person_activity["mean_daily_activity"]
)  # day-to-day instability in activity, relative to the person's own average
person_activity = person_activity[person_activity["n_valid_days"] >= MIN_VALID_DAYS]

act_path = os.path.join(os.getcwd(), "nhanes_person_activity.csv")
person_activity.to_csv(act_path, index=False)
print(f"Saved: {act_path}")
print(f"Participants with a usable activity baseline: {len(person_activity)}")


# ---------------------------------------------------------------------------
# STEP 4: Combine frailty index with activity features and model
# ---------------------------------------------------------------------------
combined = result.merge(person_activity, on="SEQN", how="inner")
print(f"\nParticipants with both frailty index and activity data: {len(combined)}")

combined["sex_female"] = (combined["sex"] == "female").astype(int)
combined["is_frail"] = (combined["frailty_category"] == "frail").astype(int)

feature_cols = [
    "age", "sex_female", "mean_daily_activity", "mean_active_minutes",
    "mean_sedentary_minutes", "activity_cv",
]
model_df = combined.dropna(subset=feature_cols + ["is_frail"])

X = model_df[feature_cols]
y = model_df["is_frail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_s, y_train)

y_pred_proba = clf.predict_proba(X_test_s)[:, 1]
y_pred = clf.predict(X_test_s)

print("\n--- Model: predicting frail vs. not frail ---")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(classification_report(y_test, y_pred))

print("Feature coefficients (standardized):")
for name, coef in zip(feature_cols, clf.coef_[0]):
    print(f"  {name:25s} {coef:+.3f}")

combined_path = os.path.join(os.getcwd(), "nhanes_frailty_and_activity.csv")
combined.to_csv(combined_path, index=False)
print(f"Saved: {combined_path}")

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=5)
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(prob_pred, prob_true, marker="o", label="Model")
ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
ax2.set_xlabel("Predicted frailty risk")
ax2.set_ylabel("Observed frailty rate")
ax2.set_title("Calibration: predicted vs. observed frailty risk")
ax2.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# STEP 5: Explainability (SHAP)
# Extends the instance-level explanation approach (LIME) used previously for
# the fall-detection model to this frailty-risk model, using SHAP, which is
# the more common choice in current clinical-ML literature.
# ---------------------------------------------------------------------------
if SHAP_AVAILABLE:
    X_test_named = pd.DataFrame(X_test_s, columns=feature_cols)

    explainer = shap.LinearExplainer(clf, X_train_s, feature_names=feature_cols)
    shap_values = explainer(X_test_named)

    # Global view: which features matter most across all test participants
    plt.figure()
    shap.summary_plot(shap_values, X_test_named, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.show()

    # Individual view: explain the single highest-predicted-risk participant
    # in the test set - the kind of per-person explanation a clinician would
    # actually want to see for one patient.
    highest_risk_idx = int(np.argmax(y_pred_proba))
    print(f"\nExplaining participant with the highest predicted frailty risk "
          f"(predicted probability = {y_pred_proba[highest_risk_idx]:.2f}):")
    plt.figure()
    shap.plots.waterfall(shap_values[highest_risk_idx], show=False)
    plt.tight_layout()
    plt.show()

    shap_summary = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    print("\nFeature importance (mean |SHAP value| across test set):")
    print(shap_summary.to_string(index=False))

    shap_path = os.path.join(os.getcwd(), "nhanes_shap_feature_importance.csv")
    shap_summary.to_csv(shap_path, index=False)
    print(f"Saved: {shap_path}")
else:
    print("\nSkipping Step 5 (SHAP explainability) - 'shap' not installed.")
