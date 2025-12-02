# ================================
#      STREAMLIT DASHBOARD
#  Customer Churn Prediction App
# ================================

import streamlit as st
import mysql.connector
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")


# --------------------------------
# DB CONNECTION
# --------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="customer-churn"
    )


# --------------------------------
# LOAD CUSTOMERS FROM DB
# --------------------------------
@st.cache_data(ttl=300)
def load_customers_from_db():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM customer_churn")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Cleaning
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"\s+", "", regex=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns and "Churn_flag" not in df.columns:
        df["Churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# --------------------------------
# FEATURE PREP FOR SINGLE PREDICT
# --------------------------------
def prepare_features_for_model(raw_df, model_columns):
    df = raw_df.copy()

    drop_cols = [c for c in ["customerID", "Churn", "Churn_flag"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    X_encoded = pd.get_dummies(df, drop_first=True)

    for col in model_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    extra = [c for c in X_encoded.columns if c not in model_columns]
    if extra:
        X_encoded = X_encoded.drop(columns=extra)

    return X_encoded[model_columns]


# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_rf.pkl")
    model_cols = joblib.load("model_columns.pkl")
    return model, model_cols


model, model_columns = load_model()


# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------
page = st.sidebar.radio(
    "📌 Navigate",
    ["EDA", "Predictions", "Feature Importance", "Single Predict"]
)


# ===========================================================
#                      E D A   P A G E
# ===========================================================
if page == "EDA":
    st.header("📊 Interactive Exploratory Data Analysis")

    df = load_customers_from_db()

    if df.empty:
        st.info("No data in `customer_churn` table.")
    else:
        # KPIs
        total = len(df)
        churn_count = df["Churn_flag"].sum()
        churn_rate = churn_count / total * 100

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Customers", f"{total:,}")
        k2.metric("Churn Count", f"{churn_count:,}")
        k3.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.markdown("---")

        # ======================
        # 1) TENURE DISTRIBUTION
        # ======================
        st.subheader("Tenure Distribution by Churn")

        fig1 = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            title="Tenure vs Churn"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("""
        **📝 Summary**  
        - Customers with **tenure under 12 months** churn the most.  
        - Long-tenure customers (50+ months) rarely churn.  
        **💡 Insight:** Focus retention on early-tenure customers.
        """)

        st.markdown("---")

        # ======================
        # 2) MONTHLY CHARGES
        # ======================
        st.subheader("Monthly Charges Distribution by Churn")

        fig2 = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            nbins=40,
            opacity=0.7,
            barmode="overlay",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **📝 Summary**  
        - Higher monthly charges → higher churn probability.  
        **💡 Insight:** Consider discounts or loyalty rewards for high-spend customers.
        """)

        st.markdown("---")

        # ======================
        # 3) CONTRACT TYPE
        # ======================
        st.subheader("Churn by Contract Type")

        fig3 = px.histogram(
            df,
            x="Contract",
            color="Churn",
            text_auto=True,
            barmode="group",
            title="Contract vs Churn"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        **📝 Summary**  
        - **Month-to-Month** customers churn the most.  
        - **One-year** and **two-year** customers are loyal.  
        **💡 Insight:** Promote longer-term contracts.
        """)

        st.markdown("---")

        # ======================
        # 4) INTERNET SERVICE
        # ======================
        st.subheader("Churn by Internet Service Type")

        fig4 = px.histogram(
            df,
            x="InternetService",
            color="Churn",
            text_auto=True,
            barmode="group",
            title="Internet Service vs Churn"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""
        **📝 Summary**  
        - **Fiber Optic** customers churn significantly more.  
        **💡 Insight:** Investigate fiber pricing or service performance.
        """)

        st.markdown("---")

        # ======================
        # 5) PAYMENT METHOD
        # ======================
        if "PaymentMethod" in df.columns:
            st.subheader("Churn by Payment Method")

            fig5 = px.histogram(
                df,
                x="PaymentMethod",
                color="Churn",
                text_auto=True,
                barmode="group",
                title="Payment Method vs Churn"
            )
            st.plotly_chart(fig5, use_container_width=True)

            st.markdown("""
            **📝 Summary**  
            - Customers paying via **Electronic Check** churn heavily.  
            **💡 Insight:** Encourage risky customers to switch to Auto-Pay.
            """)

        st.markdown("---")

        # ======================
        # 6) CORRELATION HEATMAP
        # ======================
        st.subheader("Correlation Heatmap")

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        corr = df[numeric_cols].corr()

        fig6 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Numeric Correlation Matrix"
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("""
        **📝 Summary**  
        - TotalCharges is strongly related to MonthlyCharges and tenure.  
        - SeniorCitizen has weak correlations.  
        """)


# ===========================================================
#                      P R E D I C T I O N S
# ===========================================================
elif page == "Predictions":
    st.header("📈 Predictions Overview")

    # Load predictions
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM churn_predictions ORDER BY churn_probability DESC")
        rows = cur.fetchall()
        preds = pd.DataFrame(rows)
        cur.close()
        conn.close()
    except:
        preds = pd.DataFrame()

    if preds.empty:
        st.info("No predictions available. Run `predict.py` first.")
    else:
        preds["churn_probability_percent"] = preds["churn_probability"] * 100

        total = len(preds)
        high = (preds["churn_probability"] >= 0.7).sum()
        med = ((preds["churn_probability"] >= 0.4) & (preds["churn_probability"] < 0.7)).sum()
        low = (preds["churn_probability"] < 0.4).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", f"{total:,}")
        c2.metric("High Risk (≥70%)", f"{high:,}")
        c3.metric("Medium Risk (40–69%)", f"{med:,}")
        c4.metric("Low Risk (<40%)", f"{low:,}")

        st.markdown("---")

        # Pie chart
        risk_counts = preds["risk"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        fig = px.pie(risk_counts, names="risk", values="count", hole=0.4, title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # High-risk table
        st.subheader("🔥 Top High-Risk Customers (≥70%)")
        high_df = preds[preds["churn_probability_percent"] >= 70].reset_index(drop=True)
        st.dataframe(high_df, height=300)

        st.markdown("---")

        # Full table
        st.subheader("Full Predictions Table")
        st.dataframe(preds, height=400)

        csv = preds.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")


# ===========================================================
#                F E A T U R E   I M P O R T A N C E
# ===========================================================
elif page == "Feature Importance":
    st.header("📌 Feature Importance (Random Forest)")

    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": model_columns, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(20)

    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Most Important Features"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(fi)


# ===========================================================
#                S I N G L E  P R E D I C T I O N
# ===========================================================
elif page == "Single Predict":
    st.header(" Single Customer Prediction")

    st.write("Enter customer details to get churn prediction:")

    # Input form
    customerID = st.text_input("Customer ID", "NEW001")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
    MultipleLines = st.selectbox("MultipleLines", ["No", "Yes", "Nophoneservice"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiberoptic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "Nointernetservice"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "Nointernetservice"])
    DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "Nointernetservice"])
    TechSupport = st.selectbox("TechSupport", ["Yes", "No", "Nointernetservice"])
    StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "Nointernetservice"])
    StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "Nointernetservice"])
    Contract = st.selectbox("Contract", ["Month-to-month", "Oneyear", "Twoyear"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Mailedcheck", "Electroniccheck", "Creditcard(automatic)", "Banktransfer(automatic)"])
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0)

    if st.button("Predict Churn"):
        raw = pd.DataFrame([{
            "customerID": customerID,
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract.replace(" ", ""),
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod.replace(" ", "").replace("-", ""),
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        X = prepare_features_for_model(raw, model_columns)
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        st.metric("Churn Prediction", "Yes" if pred == 1 else "No")
        st.metric("Churn Probability", f"{proba * 100:.2f}%")