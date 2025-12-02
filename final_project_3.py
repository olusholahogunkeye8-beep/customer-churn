import pandas as pd
import numpy as np
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib





#Load dataset locally
"""
data = "/Users/kelvinanowu/Desktop/DATASET1.csv"
df = pd.read_csv(data)
print(df)"""

# Connect to XAMPP MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="customer-churn"    #  database name
)

# Query the table
query = "SELECT * FROM customer_churn"

# Load directly into pandas DataFrame
df = pd.read_sql(query, conn)
conn.close()


# basic shape & types
print(df.shape)
#print(df.dtypes)
print(df.head())

# Clean whitespace + hidden characters (\r, \n, tabs, multiple spaces)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True)

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())


# Drop customerID (not useful for ML)
df = df.drop(columns=['customerID'])

# Convert Churn to 1/0
df['Churn_flag'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Verify it worked
print(df[['Churn', 'Churn_flag']].head())

# Churn distribution counts and percentage
print(df['Churn_flag'].value_counts())
print(df['Churn_flag'].value_counts(normalize=True) * 100)


"""# Analyze Churn by Tenure
sns.kdeplot(data=df, x='tenure', hue='Churn', shade=True)
plt.title("Tenure Distribution by Churn")
plt.show()
"""

"""#Analyze Monthly Charges vs Churn
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', shade=True)
plt.title("Monthly Charges Distribution by Churn")
plt.show()


#Contract Type vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()

# Internet Service vs churn
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Churn by Internet Service Type")
plt.show()"""


#------ DATA PREPARATION (Before Modeling) -------

# Separate target and features
X = df.drop(columns=['Churn', 'Churn_flag'])
y = df['Churn_flag']

# convert all string columns into numeric variables.
X_encoded = pd.get_dummies(X, drop_first=True)
y = y.fillna(y.mode()[0])
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,test_size=0.2,stratify=y,random_state=42)






               # ----------RANDOM FOREST --------------
"""rf = RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced')

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))"""


# ----- Fine Tune Random Forest to improve important metrics

# Parameter grid to search
param_grid = {
    'n_estimators': [200, 300, 500, 700],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)

# Random search setup
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=25,                  # number of combinations to try
    scoring='recall',           # focus on recall for churn class
    cv=3,                       # 3-fold cross validation
    verbose=2,
    n_jobs=-1,                  # use all CPU cores
    random_state=42
)

# Fit search
search.fit(X_train, y_train)

print("Best parameters found:")
print(search.best_params_)
"""

"""# --- Train the final…



# --- Train the final tuned Random Forest ----
best_rf = search.best_estimator_

best_rf.fit(X_train, y_train)

y_pred_best = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]

print("Tuned Random Forest Report:")
print(classification_report(y_test, y_pred_best))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_best))




               #-------------------- XGBOOST --------------------------

"""xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight= (y_train.value_counts()[0] / y_train.value_counts()[1])   # handles imbalance
)

xgb.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))
"""



#-------- Feature Importance -----------

# Get feature importances from the tuned RF model
importances = best_rf.feature_importances_

# Create a DataFrame
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df.head(15))


# Visualise the top features

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'][:15], feature_importance_df['importance'][:15])
plt.gca().invert_yaxis()  # Largest at top
plt.title("Top 15 Most Important Features — Tuned Random Forest")
plt.xlabel("Feature Importance Score")
plt.show()

# save model columns
joblib.dump(list(X_train.columns), "model_columns.pkl")
print("Saved model_columns.pkl")


# Save the tuned Random Forest model
joblib.dump(best_rf, "best_rf.pkl")

print("Model saved as best_rf.pkl")

