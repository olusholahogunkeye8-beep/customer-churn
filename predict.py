import mysql.connector
import pandas as pd
import joblib
import numpy as np
import csv

# ---------------- CONFIG ----------------
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASS = ""
MYSQL_DB   = "customer-churn"

SOURCE_TABLE = "customer_churn"     # main dataset table
PRED_TABLE   = "churn_predictions"  # prediction output table
CSV_OUTPUT   = "churn_predictions.csv"

MODEL_PKL = "best_rf.pkl"
MODEL_COLUMNS_PKL = "model_columns.pkl"

# ----------------------------------------
# Helper
# ----------------------------------------
def risk_level(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

# ----------------------------------------
# Load model + columns
# ----------------------------------------
best_rf = joblib.load(MODEL_PKL)
model_columns = joblib.load(MODEL_COLUMNS_PKL)
print("Loaded model and columns. Expected features:", len(model_columns))

# ----------------------------------------
# Connect to MySQL
# ----------------------------------------
conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASS,
    database=MYSQL_DB
)
cursor = conn.cursor(dictionary=True)

print("Reading data from table:", SOURCE_TABLE)
cursor.execute(f"SELECT * FROM {SOURCE_TABLE}")
rows = cursor.fetchall()
df_db = pd.DataFrame(rows)
print("Rows loaded:", df_db.shape)

if 'customerID' not in df_db.columns:
    raise KeyError("customerID column missing — required for output.")

# ----------------------------------------
# CLEANING (same as training)
# ----------------------------------------
for col in df_db.select_dtypes(include=['object']).columns:
    df_db[col] = df_db[col].astype(str).str.replace(r'\s+', '', regex=True)

if 'TotalCharges' in df_db.columns:
    df_db['TotalCharges'] = pd.to_numeric(df_db['TotalCharges'], errors='coerce')

# ----------------------------------------
# Feature matrix creation
# ----------------------------------------
drop_cols = [c for c in ['customerID','Churn','Churn_flag'] if c in df_db.columns]
X_all = df_db.drop(columns=drop_cols, errors='ignore')

X_encoded = pd.get_dummies(X_all, drop_first=True)

# Add missing columns
missing = [c for c in model_columns if c not in X_encoded.columns]
for c in missing:
    X_encoded[c] = 0

# Drop extras
extra = [c for c in X_encoded.columns if c not in model_columns]
if extra:
    X_encoded.drop(columns=extra, inplace=True)

# Reorder
X_encoded = X_encoded[model_columns]

print("Final feature matrix:", X_encoded.shape)

# ----------------------------------------
# Predict
# ----------------------------------------
probs = best_rf.predict_proba(X_encoded)[:,1]
preds = best_rf.predict(X_encoded)

# Build output
out = pd.DataFrame({
    "customerID": df_db['customerID'],
    "predicted_churn": preds,
    "churn_probability": probs,
})

out["risk"] = out["churn_probability"].apply(risk_level)

# Attach useful context fields
for ctx in ["tenure", "MonthlyCharges", "Contract", "InternetService"]:
    if ctx in df_db.columns:
        out[ctx] = df_db[ctx]

print(out.head())

# ----------------------------------------
# SAVE to CSV
# ----------------------------------------
out.to_csv(CSV_OUTPUT, index=False)
print("Saved CSV:", CSV_OUTPUT)

# ----------------------------------------
# SAVE to MySQL (WRITE)
# ----------------------------------------
print(f"Writing predictions to MySQL table '{PRED_TABLE}' using mysql.connector...")

# Drop and recreate table
cursor.execute(f"DROP TABLE IF EXISTS {PRED_TABLE}")
cursor.execute(f"""
    CREATE TABLE {PRED_TABLE} (
        customerID VARCHAR(100),
        predicted_churn INT,
        churn_probability FLOAT,
        risk VARCHAR(20),
        tenure INT,
        MonthlyCharges FLOAT,
        Contract VARCHAR(50),
        InternetService VARCHAR(50)
    )
""")

# Insert manually
insert_sql = f"""
INSERT INTO {PRED_TABLE} 
(customerID, predicted_churn, churn_probability, risk, tenure, MonthlyCharges, Contract, InternetService)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

for _, row in out.iterrows():
    cursor.execute(insert_sql, (
        row["customerID"],
        int(row["predicted_churn"]),
        float(row["churn_probability"]),
        row["risk"],
        int(row["tenure"]) if "tenure" in out.columns else None,
        float(row["MonthlyCharges"]) if "MonthlyCharges" in out.columns else None,
        str(row["Contract"]) if "Contract" in out.columns else None,
        str(row["InternetService"]) if "InternetService" in out.columns else None,
    ))

conn.commit()
cursor.close()
conn.close()

print("DONE — Predictions written into MySQL + CSV.")