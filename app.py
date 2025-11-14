import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Attrition Dashboard (Premium)", layout="wide")

MODEL_PATH = "attrition_model_pipeline.pkl"

@st.cache_resource
def load_artifact(path=MODEL_PATH):
    art = joblib.load(path)
    return art

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
train_columns = artifact["columns"]

# --- Helpers ---
def prepare_input(df):
    # engineered features
    if 'YearsAtCompany' in df.columns and 'Age' in df.columns:
        df['YearsAtCompany_by_Age'] = df['YearsAtCompany'] / (df['Age'] + 1)
    if 'YearsSinceLastPromotion' in df.columns:
        df['YearsSinceLastPromotion_flag'] = (df['YearsSinceLastPromotion'] > 0).astype(int)
    # encode and align
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=train_columns, fill_value=0)
    return df_enc

def predict_df(df):
    X = prepare_input(df)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)[:,1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
    return preds, probs, X

# --- UI Layout ---
st.title("🚀 Employee Attrition — Premium App")
st.markdown("Multi-page app: **Predict**, **Batch**, **Dashboard**, **Explain**")

page = st.sidebar.selectbox("Navigation", ["Single Predict", "Batch Predict", "Dashboard", "Explainability", "Download"])

# -------------------- Single Predict --------------------
if page == "Single Predict":
    st.header("Single Employee Prediction")
    with st.form("single_form"):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input("Age", 18, 65, 29)
            business_travel = st.selectbox("BusinessTravel", ["Travel_Rarely","Travel_Frequently","Non-Travel"])
            daily_rate = st.number_input("DailyRate", 0, 5000, 800)
            department = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
            distance = st.number_input("DistanceFromHome", 0, 50, 5)
            education = st.selectbox("Education", [1,2,3,4,5])
            env_sat = st.selectbox("EnvironmentSatisfaction", [1,2,3,4])
        with cols[1]:
            gender = st.selectbox("Gender", ["Male","Female"])
            job_level = st.number_input("JobLevel", 1, 10, 1)
            job_role = st.selectbox("JobRole", [
                "Sales Executive","Research Scientist","Laboratory Technician","Manager",
                "Manufacturing Director","Healthcare Representative","Sales Representative"])
            job_sat = st.selectbox("JobSatisfaction", [1,2,3,4])
            marital = st.selectbox("MaritalStatus", ["Single","Married","Divorced"])
            monthly_income = st.number_input("MonthlyIncome", 0, 100000, 2500)
        with cols[2]:
            num_comp = st.number_input("NumCompaniesWorked", 0, 50, 2)
            over_time = st.selectbox("OverTime", ["Yes","No"])
            pct_hike = st.number_input("PercentSalaryHike", 0, 100, 13)
            perf = st.selectbox("PerformanceRating", [1,2,3,4])
            rel_sat = st.selectbox("RelationshipSatisfaction", [1,2,3,4])
            stock = st.number_input("StockOptionLevel", 0, 5, 0)
            total_years = st.number_input("TotalWorkingYears", 0, 60, 3)
        submit = st.form_submit_button("Predict")
    if submit:
        sample = pd.DataFrame([{
            'Age': int(age),'BusinessTravel': business_travel, 'DailyRate': float(daily_rate),
            'Department': department, 'DistanceFromHome': int(distance),'Education': int(education),
            'EnvironmentSatisfaction': int(env_sat),'Gender': gender,'JobLevel': int(job_level),
            'JobRole': job_role,'JobSatisfaction': int(job_sat),'MaritalStatus': marital,
            'MonthlyIncome': float(monthly_income),'NumCompaniesWorked': int(num_comp),
            'OverTime': over_time,'PercentSalaryHike': int(pct_hike),'PerformanceRating': int(perf),
            'RelationshipSatisfaction': int(rel_sat),'StockOptionLevel': int(stock),'TotalWorkingYears': int(total_years)
        }])
        preds, probs, X = predict_df(sample)
        st.metric("Attrition (Yes=1)", int(preds[0]))
        st.write(f"Probability of leaving: **{probs[0]:.4f}**")
        # show top features if possible
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=train_columns).sort_values(ascending=False).head(10)
            st.subheader("Top features (local)")
            st.table(importances.rename_axis("feature").reset_index(name="importance"))

# -------------------- Batch Predict --------------------
elif page == "Batch Predict":
    st.header("Batch Prediction (CSV upload)")
    st.markdown("Upload CSV with columns similar to training data (can have extra columns).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            st.write("Preview (first 5 rows)")
            st.dataframe(df_batch.head())
            preds, probs, X = predict_df(df_batch)
            df_out = df_batch.copy()
            df_out['Attrition_Pred'] = preds
            df_out['Attrition_Prob'] = probs
            st.success("Batch prediction done!")
            st.dataframe(df_out.head())
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# -------------------- Dashboard --------------------
elif page == "Dashboard":
    st.header("Dataset-level Dashboard (Synthetic)")
    st.markdown("Quick KPIs and distributions based on training artifact columns.")
    # Try to reconstruct a simple synthetic dataset summary using train_columns
    # We do not have original training dataframe here; provide simple visuals from model columns
    st.subheader("Top model features (global)")
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=train_columns).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=fi.values, y=fi.index, ax=ax)
        ax.set_title("Top 20 Feature Importances")
        st.pyplot(fig)
    else:
        st.info("Model doesn't provide feature_importances_.")

# -------------------- Explainability --------------------
elif page == "Explainability":
    st.header("Explainability (SHAP, if available)")
    if st.button("Run SHAP on sample"):
        try:
            import shap
            # prepare a small sample from zero/ones using train_columns to pass shape
            blank = pd.DataFrame([dict((c,0) for c in train_columns)])
            sample_X = prepare_for_shap = blank.values
            explainer = shap.TreeExplainer(model)
            # Use small random sample: create random data around zeros
            rnd = np.random.RandomState(42)
            sample_for_shap = np.vstack([rnd.normal(size=len(train_columns)) for _ in range(50)])
            shap_values = explainer.shap_values(sample_for_shap)
            st.success("SHAP computed. Displaying summary plot...")
            # Show summary with matplotlib
            try:
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, sample_for_shap, show=False)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.image(buf)
            except Exception as e:
                st.error(f"SHAP plotting failed: {e}")
        except Exception as e:
            st.error(f"SHAP not available or failed: {e}")

# -------------------- Download --------------------
elif page == "Download":
    st.header("Download Project & Model")
    st.markdown("You can download the model (`.pkl`) and instructions here.")
    with open(MODEL_PATH, "rb") as f:
        bytes_data = f.read()
    st.download_button("Download model (.pkl)", data=bytes_data, file_name="attrition_model_pipeline.pkl", mime="application/octet-stream")
    st.markdown("**README** is included in the repo for deployment instructions.")
