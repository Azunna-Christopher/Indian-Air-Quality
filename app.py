import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------
st.set_page_config(
    page_title="EcoVis | Air Quality Forecaster",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1 { color: #2E86C1; }
    div.stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------
# 2. LOAD DATA & MODELS
# -------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_merged_dataset.csv")

@st.cache_resource
def load_models():
    # Note: In Colab, we moved files to 'models/' in Step 1
    rf = joblib.load('models/aqi_model.pkl')
    knn = joblib.load('models/knn_model.pkl')
    return rf, knn

try:
    df = load_data()
    rf_model, knn_model = load_models()
except FileNotFoundError as e:
    st.error(f"❌ System Error: {e}. Ensure 'preprocessed_merged_dataset.csv' and 'models/' folder exist.")
    st.stop()

# -------------------------------------------
# 3. SIDEBAR NAVIGATION
# -------------------------------------------
st.sidebar.title("EcoVis System")
st.sidebar.markdown("---")
page = st.sidebar.radio("📍 Navigation", ["Dashboard Overview", "Deep Dive Analytics", "AI Forecaster"])
st.sidebar.markdown("---")

if page == "AI Forecaster":
    st.sidebar.subheader("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Select Algorithm:", ["Random Forest", "KNN Regression"])

st.sidebar.caption("Data Source: CPCB India (2015-2020)")

# -------------------------------------------
# 4. DASHBOARD OVERVIEW
# -------------------------------------------
if page == "Dashboard Overview":
    st.title("🌿 Environmental Dashboard")
    st.markdown("### Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Records", f"{len(df):,}")
    with c2: st.metric("Cities", f"{df['City'].nunique()}")
    with c3: st.metric("Avg AQI", f"{df['AQI'].mean():.0f}")
    with c4: st.metric("Pollutants", "11 Types")

    st.markdown("---")
    st.subheader("📋 Recent Data Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------
# 5. ANALYTICS
# -------------------------------------------
elif page == "Deep Dive Analytics":
    st.title("📊 Pollution Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Correlation Matrix")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### AQI Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(df['AQI'], bins=40, kde=True, color='#E67E22', ax=ax2)
        st.pyplot(fig2)

# -------------------------------------------
# 6. AI FORECASTER
# -------------------------------------------
elif page == "AI Forecaster":
    st.title("🤖 AI Forecaster")
    
    with st.container():
        st.subheader("🧪 Pollutant Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            pm25 = st.number_input("PM2.5", 0.0, 1000.0, 50.0)
            pm10 = st.number_input("PM10", 0.0, 1000.0, 100.0)
            no = st.number_input("NO", 0.0, 500.0, 20.0)
            no2 = st.number_input("NO2", 0.0, 500.0, 30.0)
        with c2:
            nox = st.number_input("NOx", 0.0, 500.0, 40.0)
            nh3 = st.number_input("NH3", 0.0, 500.0, 25.0)
            co = st.number_input("CO", 0.0, 50.0, 1.0)
            so2 = st.number_input("SO2", 0.0, 200.0, 10.0)
        with c3:
            o3 = st.number_input("O3", 0.0, 300.0, 40.0)
            benzene = st.number_input("Benzene", 0.0, 200.0, 2.0)
            toluene = st.number_input("Toluene", 0.0, 200.0, 5.0)

    st.markdown("---")
    
    if st.button("🚀 Generate Prediction", use_container_width=True):
        input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene]],
                                  columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene'])
        
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)[0]
        else:
            prediction = knn_model.predict(input_data)[0]

        st.markdown("### 🎯 Prediction Results")
        
        # Color Logic
        if prediction <= 50: status, color = "Good", "green"
        elif prediction <= 100: status, color = "Satisfactory", "blue"
        elif prediction <= 200: status, color = "Moderate", "orange"
        elif prediction <= 300: status, color = "Poor", "red"
        else: status, color = "Hazardous", "darkred"

        c_res1, c_res2 = st.columns([1, 2])
        with c_res1: st.metric("Predicted AQI", f"{prediction:.2f}")
        with c_res2: st.info(f"**Category:** {status}")
          
