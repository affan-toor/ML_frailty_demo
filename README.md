# Frailty risk from open epidemiological and wearable-activity data (NHANES)

A small exploratory pipeline that estimates frailty risk in older adults by
combining two kinds of open data: survey/clinical (register-style) data and
wrist-accelerometer (wearable) data, from the same NHANES participants.

This was built as a personal proof-of-concept exploring how register data
and daily-life wearable activity data could be combined for early frailty
identification, using only open US data. It does **not** use HUNT data,
Norwegian register data, or any data from a specific research project —
it is a methods demonstration on public NHANES data only.

## What it does

1. **Frailty index** — builds a Rockwood-style deficit-accumulation frailty
   index for adults aged 60+ from NHANES 2013-2014 questionnaire, exam, and
   lab data (chronic conditions, diabetes, hypertension/cholesterol,
   self-rated health, PHQ-9 depression items, obesity).
2. **Wearable activity features** — for the same participants, derives
   personal activity-pattern features from 7-9 days of wrist-accelerometer
   data (mean daily activity, active/sedentary minutes, and day-to-day
   coefficient of variation as a measure of behavioural instability).
3. **Combined model** — fits a logistic regression predicting frail vs.
   not-frail status from age, sex, and the activity features, with
   ROC-AUC, a classification report, and a calibration plot.
4. **Explainability** — uses SHAP to show which features drive the model's
   predictions overall, and to explain one individual's predicted risk in
   detail (extending an earlier LIME-based explainability approach used in
   a wearable-sensor fall-detection model).

## Why this combination

The frailty index reflects accumulated health deficits (the "register"
side); the activity features reflect how a person is actually moving
day to day (the "wearable" side). Looking at each person's activity
relative to their *own* multi-day baseline, rather than a single
population-wide cut-off, follows the same contextual/personal-baseline
logic used in anomaly-detection work on continuous sensor data — applied
here to a new domain and a new, open dataset.

## Data sources and citations

**NHANES 2013-2014 (register/questionnaire data)**
Centers for Disease Control and Prevention (CDC), National Center for
Health Statistics. National Health and Nutrition Examination Survey
(NHANES), 2013-2014. Available at:
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013

Files used: `DEMO_H`, `MCQ_H`, `DIQ_H`, `BPQ_H`, `HUQ_H`, `DPQ_H`, `BMX_H`.
Variable definitions: see each file's codebook page at
`https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/<FILE>.htm`.

**Wrist-accelerometer wearable data**
"Minute level step counts and physical activity data from the National
Health and Nutrition Examination Survey (NHANES) 2011-2014" (version
1.0.2), PhysioNet. Project page:
https://physionet.org/content/minute-level-step-count-nhanes/1.0.2/
*(Please copy the exact author list and citation shown in the "Cite"
panel on that project page before publishing — it was not reproduced
here to avoid misattributing authorship.)*

Also cite the standard PhysioNet reference:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C.,
Mark, R., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E.
(2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new
research resource for complex physiologic signals. *Circulation*,
101(23), e215–e220.

**Wear/sleep/nonwear prediction method**
Method behind the `PAXPREDM` variable used in this project — see:
"Detecting Sleep and Nonwear in 24-h Wrist Accelerometer Data from the
National Health and Nutrition Examination Survey."
https://pubmed.ncbi.nlm.nih.gov/36007161/

**Frailty index methodology**
Searle, S. D., Mitnitski, A., Gahbauer, E. A., Gill, T. M., & Rockwood,
K. (2008). A standard procedure for creating a frailty index.
*BMC Geriatrics*, 8, 24.

**SHAP**
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
model predictions. *Advances in Neural Information Processing Systems*, 30.

## How to run

Requires Python 3.10+ with pandas, numpy, requests, matplotlib,
scikit-learn, and shap (all but shap come with a standard Anaconda
install; install shap with `pip install shap` or
`conda install -c conda-forge shap`).

```
python build_frailty_index_nhanes.py
```

The first run downloads and caches the required NHANES and PhysioNet
files locally (a few hundred MB total); later runs reuse the cache.

## Outputs

- `nhanes_frailty_index.csv` — one row per participant: age, sex, frailty
  index, frailty category.
- `nhanes_person_activity.csv` — one row per participant: personal
  wearable-activity summary features.
- `nhanes_frailty_and_activity.csv` — the two combined.
- `nhanes_shap_feature_importance.csv` — mean absolute SHAP value per
  feature.
- Plots: frailty index distribution, frailty index by sex, model
  calibration curve, SHAP summary plot, one individual SHAP waterfall plot.

## Limitations

This is a personal methods demonstration, not a validated clinical tool:

- NHANES is a US general-population survey, not an ageing/frailty cohort,
  so prevalence and associations here will not match a geriatric
  population directly.
- The deficit list (~24 items) is smaller than the 30-53 items used in
  many published frailty indices, chosen for speed of implementation.
- Accelerometer coverage in NHANES is lower for older adults, so the
  final modelling sample is modest in size.
- The activity-vs-frailty association shown here is exploratory and not
  intended to support any clinical or individual-level decision.

Disclaimer: This demo was built with the help of Claude to build understanding of the problem domain and get hands-on practice on application of ML on frailty data domain.
