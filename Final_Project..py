from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sqlite3
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r"C:\Users\USER\Downloads\Final_Project\customer_churn_dataset-training-master (1).csv")

# ----------------------------
# CONFIG - update if needed
# ----------------------------
DATA_PATH = Path(r"C:\Users\USER\Downloads\Final_Project\customer_churn_dataset-training-master (1).csv")
OUT_DIR = Path.cwd() / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filenames
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "churn_pipeline.joblib"
PPTX_PATH = OUT_DIR / "churn_presentation.pptx"
SQLITE_PATH = OUT_DIR / "churn_data.db"
SUMMARY_PATH = OUT_DIR / "summary_report.txt"

# ----------------------------
# 1. Load & validate file
# ----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV file not found at: {DATA_PATH}")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------
# 2. Basic cleaning
# ----------------------------
df_clean = df.copy()

# Remove leading/trailing spaces in column names & standardize
df_clean.columns = [c.strip().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") for c in df_clean.columns]

# If CustomerID-like column exists rename and drop if needed
for possible_id in ["CustomerID", "customerID", "customer_id", "Id", "id"]:
    if possible_id in df_clean.columns:
        print("Dropping identifier column:", possible_id)
        df_clean = df_clean.drop(columns=[possible_id])
        break

# Inspect dtypes
print("Dtypes before conversion:")
print(df_clean.dtypes)

# Example: ensure 'Churn' exists and is numeric 0/1
if "Churn" not in df_clean.columns:
    # try common alternatives
    alternatives = [c for c in df_clean.columns if "churn" in c.lower()]
    if alternatives:
        print("Using column as Churn:", alternatives[0])
        df_clean = df_clean.rename(columns={alternatives[0]: "Churn"})
    else:
        raise KeyError("No 'Churn' column found. Please ensure dataset has a churn indicator.")

# Normalize Churn values to binary 0/1
def normalize_churn(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and x in (0,1):
        return int(x)
    x = str(x).strip().lower()
    if x in ("yes","y","true","1","churn"):
        return 1
    if x in ("no","n","false","0","not churn"):
        return 0
    try:
        val = float(x)
        return int(val != 0)
    except:
        return np.nan

df_clean["Churn"] = df_clean["Churn"].apply(normalize_churn)

# Fill missing for numeric -> median. For categorical -> mode.
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
# ensure Churn is excluded from features
if "Churn" in num_cols:
    num_cols.remove("Churn")

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# Fill numeric NaNs
for c in num_cols:
    median = df_clean[c].median()
    df_clean[c] = df_clean[c].fillna(median)

# Fill categorical NaNs
for c in cat_cols:
    mode = df_clean[c].mode()
    if not mode.empty:
        df_clean[c] = df_clean[c].fillna(mode.iloc[0])
    else:
        df_clean[c] = df_clean[c].fillna("Unknown")

# If Churn still has NaNs, drop those rows (or you can impute/assume non-churn)
missing_churn = df_clean["Churn"].isna().sum()
if missing_churn > 0:
    print(f"Dropping {missing_churn} rows with unknown Churn")
    df_clean = df_clean.dropna(subset=["Churn"])
df_clean["Churn"] = df_clean["Churn"].astype(int)

print("Shape after cleaning:", df_clean.shape)

# ----------------------------
# 3. Exploratory Visualizations
# ----------------------------
sns.set(style="whitegrid")

# Age distribution (if present)
if "Age" in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df_clean["Age"], bins=20, kde=True)
    plt.title("Age distribution")
    fn = PLOTS_DIR / "age_distribution.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print("Saved:", fn)

# Tenure vs Churn rate (if Tenure exists)
if "Tenure" in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Churn", y="Tenure", data=df_clean)
    plt.title("Tenure by Churn")
    fn = PLOTS_DIR / "tenure_by_churn.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print("Saved:", fn)

# Total_Spend or similar check
spend_col_candidates = [c for c in df_clean.columns if "spend" in c.lower() or "total" in c.lower() and "spend" in c.lower()]
if spend_col_candidates:
    spend_col = spend_col_candidates[0]
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="Tenure" if "Tenure" in df_clean.columns else df_clean.index, y=spend_col, hue="Churn", data=df_clean, alpha=0.7)
    plt.title(f"{spend_col} vs Tenure colored by Churn")
    fn = PLOTS_DIR / "spend_vs_tenure.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print("Saved:", fn)

# Correlation heatmap for numeric columns
if len(num_cols) >= 2:
    plt.figure(figsize=(8,6))
    corr = df_clean[num_cols + ["Churn"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    fn = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print("Saved:", fn)

# ----------------------------
# 4. Prepare features & pipeline
# ----------------------------
X = df_clean.drop(columns=["Churn"])
y = df_clean["Churn"]

# Recompute numeric/categorical from X
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Final numeric:", num_features)
print("Final categorical:", cat_features)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features)
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# ----------------------------
# 5. Train / Evaluate
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training size:", X_train.shape)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

print("Classification report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(pipeline, MODEL_PATH)
print("Saved model to:", MODEL_PATH)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
fn = PLOTS_DIR / "confusion_matrix.png"
plt.savefig(fn, bbox_inches="tight")
plt.close()
print("Saved:", fn)

# Feature importance (approx) - requires feature names from preprocessor
try:
    # get transformed feature names
    cat_encoder = pipeline.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    feat_names = num_features + cat_names
    importances = pipeline.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]
    top_feats = [feat_names[i] for i in top_idx]
    top_imps = importances[top_idx]

    plt.figure(figsize=(8,6))
    sns.barplot(x=top_imps, y=top_feats)
    plt.title("Top 15 feature importances")
    fn = PLOTS_DIR / "feature_importances.png"
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    print("Saved:", fn)
except Exception as e:
    print("Could not compute feature importances (likely due to encoding). Error:", e)

# Save predictions to CSV
pred_df = X_test.copy()
pred_df["actual_churn"] = y_test.values
pred_df["predicted_churn"] = y_pred
pred_df["predicted_proba"] = y_proba
pred_out = OUT_DIR / "predictions_sample.csv"
pred_df.to_csv(pred_out, index=False)
print("Saved predictions sample to:", pred_out)

# ----------------------------
# 6. Export to SQLite & SQL queries
# ----------------------------
conn = sqlite3.connect(SQLITE_PATH)
df_clean.to_sql("churn_data", conn, if_exists="replace", index=False)
print("Saved dataframe to SQLite DB:", SQLITE_PATH)

# Write a helper SQL file with useful queries
sql_text = f"""
-- Useful SQL queries for churn_data table (SQLite)
-- Table: churn_data

-- Count churn vs non-churn
SELECT Churn, COUNT(*) AS cnt FROM churn_data GROUP BY Churn;

-- Average spend by churn
SELECT AVG(Total_Spend) as avg_spend, Churn
FROM churn_data
GROUP BY Churn;

-- Top 50 high-risk customers (example risk metric)
SELECT *, (COALESCE(Payment_Delay,0) * 0.6 + COALESCE(Support_Calls,0) * 0.4) as risk_score
FROM churn_data
ORDER BY risk_score DESC
LIMIT 50;
"""
sql_file = OUT_DIR / "churn_queries.sql"
sql_file.write_text(sql_text)
print("Saved helpful SQL queries to:", sql_file)

# ----------------------------
# 7. Generate PowerPoint summary
# ----------------------------
prs = Presentation()
# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Churn Prediction - Quick Summary"
if slide.placeholders:
    try:
        slide.placeholders[1].text = f"Rows: {df_clean.shape[0]} | Columns: {df_clean.shape[1]}"
    except:
        pass

# Add slide: dataset & preprocessing steps
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Dataset & Preprocessing"
tf = slide.shapes.placeholders[1].text_frame
tf.text = "Preprocessing steps performed:"
for s in [
    "Dropped identifier columns",
    "Filled numeric NaNs with median",
    "Filled categorical NaNs with mode",
    "One-hot encoding for categoricals, scaling for numerics",
    "Train/test split 80/20"
]:
    p = tf.add_paragraph()
    p.text = "- " + s

# Add slide: key plots (if exist)
slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank
left = Inches(0.5)
top = Inches(0.5)
# try to place multiple images (if they exist)
img_paths = [
    PLOTS_DIR / "age_distribution.png",
    PLOTS_DIR / "tenure_by_churn.png",
    PLOTS_DIR / "spend_vs_tenure.png",
    PLOTS_DIR / "correlation_heatmap.png",
    PLOTS_DIR / "feature_importances.png",
    PLOTS_DIR / "confusion_matrix.png"
]
x = left
y = top
w = Inches(4.5)
placed = 0
for pimg in img_paths:
    if pimg.exists():
        try:
            slide.shapes.add_picture(str(pimg), x, y, width=w)
            x = x + w + Inches(0.2)
            placed += 1
            if placed % 2 == 0:
                x = left
                y = y + Inches(3.0)
        except Exception as e:
            print("Could not add image to pptx:", pimg, e)

# Add slide: model results
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Model Results"
tf = slide.shapes.placeholders[1].text_frame
tf.text = "RandomForest quick evaluation (test set):"
p = tf.add_paragraph()
p.text = f"Rows (after cleaning): {df_clean.shape[0]}"
p = tf.add_paragraph()
p.text = f"Classification report (check console for full):"
# add short metrics
from sklearn.metrics import precision_score, recall_score, f1_score
p = tf.add_paragraph()
p.text = f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}"
p = tf.add_paragraph()
p.text = f"Accuracy (approx): {(y_pred == y_test).mean():.3f}"

# Save PPTX
prs.save(PPTX_PATH)
print("Saved presentation to:", PPTX_PATH)

# ----------------------------
# 8. Summary report
# ----------------------------
summary = f"""
Churn Prediction project summary
Input file: {DATA_PATH}
Rows after cleaning: {df_clean.shape[0]}
Columns: {df_clean.shape[1]}

Numeric features: {num_features}
Categorical features: {cat_features}

Model: RandomForest (n_estimators=200)
ROC AUC (test): {roc_auc_score(y_test, y_proba):.4f}

Generated files:
- Plots directory: {PLOTS_DIR}
- Model (joblib): {MODEL_PATH}
- Predictions sample CSV: {pred_out}
- SQLite DB: {SQLITE_PATH}
- SQL queries: {sql_file}
- Presentation: {PPTX_PATH}
"""

Path(SUMMARY_PATH).write_text(summary)
print("Saved summary to:", SUMMARY_PATH)

print("\nAll done. Check the 'outputs' folder for artifacts.")
