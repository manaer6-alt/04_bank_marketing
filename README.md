# 04_bank_marketing

## Project goal

This project is part of **ML Projects Batch 01**.

The goal is to build an honest binary classification model for a bank marketing campaign.

The model predicts whether a client is likely to subscribe to a term deposit before the final outcome of the marketing contact is known.

The main learning focus:

- binary classification;
- mixed tabular data;
- categorical and numerical preprocessing;
- class probability interpretation;
- precision / recall trade-off;
- ROC-AUC and PR-AUC;
- threshold analysis;
- leakage control.

## Dataset

Dataset: Bank Marketing Dataset

Source: Kaggle  
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

Raw file used:

```text
data/raw/bank.csv

The raw dataset is not tracked by Git.

Dataset shape
Rows: 11162
Columns: 17
Target

Target column:

deposit

Target mapping:

no  -> 0
yes -> 1

Positive class:

yes / 1

Target distribution:

no  = 5873 / 52.62%
yes = 5289 / 47.38%

There is no severe class imbalance, but the project is still a business classification problem where precision and recall matter more than accuracy alone.

Task framing

The main scenario is pre-contact prediction.

The model is intended to help prioritize clients before the final outcome of a marketing contact is known.

This means the model should only use information available before the contact decision.

Leakage control

The column duration was excluded from all model features.

Reason:

duration is the call duration. It is only known during or after the call.

For a pre-contact prediction scenario, using duration would leak future information into the model and produce unrealistically optimistic metrics.

Excluded columns:

deposit
duration

Other leakage controls:

X_test was reserved as a holdout set and evaluated only once.
Threshold tuning was performed only on train-only out-of-fold predictions.
No threshold change was made after final test evaluation.
No alternative models were evaluated on X_test after seeing final test metrics.
All preprocessing was kept inside Pipeline / ColumnTransformer.
Features

Final numeric features:

age
balance
day
campaign
pdays
previous

Final categorical features:

job
marital
education
default
housing
loan
contact
month
poutcome

The literal value unknown was kept as a categorical level and was not automatically converted to missing values.

Train/test split

A stratified train/test split was used:

test_size = 0.2
random_state = 42
stratify = y

Split sizes:

X_train: 8929 rows, 15 features
X_test:  2233 rows, 15 features

Target distribution was preserved:

Full:
no  = 52.62%
yes = 47.38%

Train:
no  = 52.62%
yes = 47.38%

Test:
no  = 52.62%
yes = 47.38%
Preprocessing

All preprocessing was placed inside scikit-learn Pipeline / ColumnTransformer.

Numeric preprocessing:

SimpleImputer(strategy="median")
StandardScaler()

Categorical preprocessing:

SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
Model development summary
Baseline models

Initial baseline models were evaluated with 5-fold StratifiedKFold cross-validation on X_train only.

Models included:

DummyClassifier(strategy="most_frequent")
LogisticRegression
LogisticRegression(class_weight="balanced")
DecisionTreeClassifier(max_depth=3)

Best simple baseline:

LogisticRegression(class_weight="balanced")

CV metrics:

accuracy  ≈ 0.7018
precision ≈ 0.7117
recall    ≈ 0.6228
F1        ≈ 0.6641
ROC-AUC   ≈ 0.7632
PR-AUC    ≈ 0.7634
Model family comparison

Stronger model families were compared using 5-fold StratifiedKFold cross-validation on X_train only.

Models included:

RandomForestClassifier
ExtraTreesClassifier
GradientBoostingClassifier
HistGradientBoostingClassifier

Best overall candidate:

HistGradientBoostingClassifier

CV metrics:

accuracy  ≈ 0.7349
precision ≈ 0.7689
recall    ≈ 0.6299
F1        ≈ 0.6923
ROC-AUC   ≈ 0.7896
PR-AUC    ≈ 0.7842

Best model by recall at default threshold:

RandomForestClassifier
recall ≈ 0.6445
Threshold analysis

Threshold analysis was performed only with 5-fold out-of-fold predicted probabilities on X_train.

No threshold was selected using X_test.

Best OOF-F1 threshold policy:

Model: HistGradientBoostingClassifier
Threshold: 0.35

Stage 5 OOF metrics at threshold 0.35:

accuracy  ≈ 0.6972
precision ≈ 0.6515
recall    ≈ 0.7762
F1        ≈ 0.7084

Interpretation:

Lowering the threshold from 0.50 to 0.35 increases recall and reduces false negatives, but increases false positives and lowers precision.

This is appropriate if the business prefers catching more potential deposit subscribers and can tolerate additional false-positive contacts.

Final model

Final frozen model:

HistGradientBoostingClassifier(random_state=42)

Final frozen threshold:

0.35

Prediction policy:

predict 1 if predicted probability >= 0.35
predict 0 otherwise

The final pipeline was fitted only on X_train / y_train.

The holdout test set was evaluated exactly once.

Final test metrics

Final test metrics at threshold 0.35:

accuracy  = 0.7035
precision = 0.6597
recall    = 0.7732
F1        = 0.7119
ROC-AUC   = 0.7890
PR-AUC    = 0.7900
Confusion matrix
TN = 753
FP = 422
FN = 240
TP = 818

Matrix:

	Predicted no	Predicted yes
Actual no	753	422
Actual yes	240	818

Interpretation:

818 real deposit subscribers were correctly identified.
240 real deposit subscribers were missed.
422 non-subscribers were incorrectly predicted as positive.
753 non-subscribers were correctly rejected.
OOF vs final test comparison

Stage 5 OOF at threshold 0.35:

accuracy  = 0.6972
precision = 0.6515
recall    = 0.7762
F1        = 0.7084

Stage 6 final test at threshold 0.35:

accuracy  = 0.7035
precision = 0.6597
recall    = 0.7732
F1        = 0.7119

Delta test minus OOF:

accuracy  +0.0063
precision +0.0082
recall    -0.0030
F1        +0.0035

The final test metrics aligned closely with train-only OOF expectations.

Final model artifact

The final model bundle was saved locally to:

models/bank_marketing_final_bundle.joblib

The bundle includes:

fitted pipeline;
threshold;
target mapping;
positive class;
numeric feature list;
categorical feature list;
excluded columns;
final test metrics;
confusion matrix;
metadata.

The model artifact is intentionally ignored by Git.

Limitations
The threshold 0.35 was selected using OOF F1, not a real monetary cost matrix.
False positives may be expensive if sales capacity is limited.
False negatives may be expensive if missed subscribers are highly valuable.
The dataset includes day and month, but not a full timestamp or year, so a reliable chronological validation split was not possible.
The model is trained on historical campaign data and may not generalize perfectly to future campaigns if customer behavior, campaign strategy, or economic conditions change.
The model is designed for pre-contact prioritization. It intentionally excludes duration, so it may score lower than leakage-contaminated models but is more realistic.
Project structure
04_bank_marketing/
├── data/
│   └── raw/
│       └── bank.csv                  # not tracked by Git
├── models/
│   └── bank_marketing_final_bundle.joblib  # not tracked by Git
├── notebooks/
│   ├── 00_dataset_audit.ipynb
│   ├── 01_initial_eda.ipynb
│   ├── 02_baseline_classification.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_threshold_and_refinement.ipynb
│   ├── 05_final_evaluation.ipynb
│   └── 06_save_final_pipeline.ipynb
├── reports/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
Reproduction steps
1. Clone repository
git clone https://github.com/manaer6-alt/04_bank_marketing.git
cd 04_bank_marketing
2. Create virtual environment
python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Add raw dataset

Download the Bank Marketing Dataset from Kaggle and place the CSV file here:

data/raw/bank.csv
5. Run notebooks in order
notebooks/00_dataset_audit.ipynb
notebooks/01_initial_eda.ipynb
notebooks/02_baseline_classification.ipynb
notebooks/03_model_comparison.ipynb
notebooks/04_threshold_and_refinement.ipynb
notebooks/05_final_evaluation.ipynb
notebooks/06_save_final_pipeline.ipynb

Important:

05_final_evaluation.ipynb evaluates the holdout test set. It should be treated as a final evaluation notebook, not as a tuning notebook.

Final status

Final model selected and evaluated:

HistGradientBoostingClassifier(random_state=42)
threshold = 0.35

Final test F1:

0.7119

Final test recall:

0.7732

Final test precision:

0.6597

"@ | Out-File -Encoding utf8 README.md


---

## 4. Что должно получиться

В корне проекта должен появиться:

```text
README.md