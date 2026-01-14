import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------
st.set_page_config(
    page_title="EcoVis | Air Quality Forecaster",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED for Contrast Issues
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f4f6f9; }
    
    /* Metric Cards - White Background with FORCED BLACK TEXT */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        color: #000000; /* Force text to black */
    }
    
    /* Force specific label colors inside metrics to ensure visibility */
    div[data-testid="stMetric"] label { color: #333333 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #000000 !important; }

    /* Titles */
    h1, h2, h3 { color: #2E86C1; }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1B4F72;
        color: white;
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
    # Looks inside 'models/' folder (Standard GitHub structure)
    rf = joblib.load('models/aqi_model.pkl')
    knn = joblib.load('models/knn_model.pkl')
    return rf, knn

try:
    df = load_data()
    rf_model, knn_model = load_models()
except FileNotFoundError as e:
    st.error(f"❌ System Error: {e}. Ensure CSV is in root and models are in 'models/' folder.")
    st.stop()

# -------------------------------------------
# 3. SIDEBAR NAVIGATION & FILTERS
# -------------------------------------------
st.sidebar.title("EcoVis System")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3214/3214170.png", width=50)
st.sidebar.markdown("---")

# Main Page Navigation
page = st.sidebar.radio("📍 Navigation", ["Dashboard Overview", "Deep Dive Analytics", "AI Forecaster"])

st.sidebar.markdown("---")

# --- GLOBAL FILTER: CITY SELECTION ---
# This applies to Dashboard and Analytics
st.sidebar.subheader("🌍 Location Filter")
city_list = sorted(df['City'].unique())
# Add 'All Cities' option
selected_city = st.sidebar.selectbox("Select Region:", ["All Cities"] + city_list)

# Filter the dataframe based on selection
if selected_city == "All Cities":
    filtered_df = df
    location_title = "All India"
else:
    filtered_df = df[df['City'] == selected_city]
    location_title = selected_city

st.sidebar.markdown("---")

# Model Settings (Only for Forecaster)
if page == "AI Forecaster":
    st.sidebar.subheader("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Select Algorithm:", ["Random Forest", "KNN Regression"])
    st.sidebar.info("Random Forest is recommended for higher accuracy.")

st.sidebar.caption("Data Source: CPCB India (2015-2020)")

# -------------------------------------------
# 4. DASHBOARD OVERVIEW
# -------------------------------------------
if page == "Dashboard Overview":
    st.title(f"🌿 Dashboard: {location_title}")
    st.markdown("### Executive Summary")

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Records", f"{len(filtered_df):,}")
    
    with c2: 
        if selected_city == "All Cities":
            st.metric("Cities Monitored", f"{filtered_df['City'].nunique()}")
        else:
            st.metric("City Ranking", "Active") # Placeholder for single city

    with c3: 
        avg_aqi = filtered_df['AQI'].mean()
        delta_val = avg_aqi - 100 # Compare to safe limit
        st.metric("Average AQI", f"{avg_aqi:.0f}", delta=f"{delta_val:.0f} vs Safe", delta_color="inverse")

    with c4: 
        # Find dominant pollutant for this selection
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
        dominant = filtered_df[pollutants].mean().idxmax()
        st.metric("Dominant Pollutant", dominant)

    st.markdown("---")
    
    # New Section: Time Series Trend for Selected Location
    st.subheader(f"📈 AQI Trend over Time ({location_title})")
    
    # Group by Date (ensure Date is parsed)
    if 'Date' in filtered_df.columns:
        # Create a smaller DF for plotting to be fast
        chart_df = filtered_df.copy()
        chart_df['Date'] = pd.to_datetime(chart_df['Date'])
        chart_data = chart_df.groupby('Date')['AQI'].mean()
        st.line_chart(chart_data, color="#2E86C1")
    else:
        st.warning("Date column not found for trend analysis.")

    st.subheader("📋 Recent Data Snapshot")
    st.dataframe(filtered_df.head(10), use_container_width=True)

# -------------------------------------------
# 5. DEEP DIVE ANALYTICS
# -------------------------------------------
elif page == "Deep Dive Analytics":
    st.title(f"📊 Analytics: {location_title}")
    
    # User Selection for Chart Type
    chart_type = st.selectbox(
        "Select Visualization Type:",
        ["Pollutant Breakdown (Bar Chart)", "AQI Distribution (Histogram)", "Correlation Heatmap", "Yearly Trends"]
    )
    
    st.markdown("---")

    # 1. Pollutant Breakdown
    if chart_type == "Pollutant Breakdown (Bar Chart)":
        st.subheader(f"Average Pollutant Levels in {location_title}")
        pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
        
        # Calculate means
        avg_levels = filtered_df[pollutants].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=avg_levels.values, y=avg_levels.index, palette="magma", ax=ax)
        ax.set_xlabel("Concentration (µg/m³)")
        st.pyplot(fig)

    # 2. Distribution
    elif chart_type == "AQI Distribution (Histogram)":
        st.subheader("Frequency of AQI Levels")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['AQI'], bins=40, kde=True, color='#E67E22', ax=ax)
        ax.set_xlabel("Air Quality Index (AQI)")
        st.pyplot(fig)

    # 3. Correlation
    elif chart_type == "Correlation Heatmap":
        st.subheader("Pollutant Correlation Matrix")
        st.write("How do different chemicals relate to each other in this region?")
        numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # 4. Yearly Trends (Boxplot)
    elif chart_type == "Yearly Trends":
        st.subheader("AQI Variance by Year")
        if 'Year_bin' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Year_bin', y='AQI', data=filtered_df, palette="viridis", ax=ax)
            st.pyplot(fig)
        else:
            st.error("Year column not found in dataset.")

# -------------------------------------------
# 6. AI FORECASTER
# -------------------------------------------
elif page == "AI Forecaster":
    st.title("🤖 Real-Time AI Forecaster")
    st.write("Enter specific pollutant values to predict the Air Quality Index.")
    
    # Using a white container for inputs
    with st.container():
        st.markdown("#### 🧪 Pollutant Concentrations")
        
        # 3 Column Layout
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
    
    # Prediction Action
    if st.button("🚀 Generate Prediction", use_container_width=True):
        input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene]],
                                  columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene'])
        
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)[0]
        else:
            prediction = knn_model.predict(input_data)[0]

        # Results Section
        st.markdown("### 🎯 Prediction Results")
        
        # Color Logic
        if prediction <= 50: status, color = "Good", "green"
        elif prediction <= 100: status, color = "Satisfactory", "blue"
        elif prediction <= 200: status, color = "Moderate", "orange"
        elif prediction <= 300: status, color = "Poor", "red"
        else: status, color = "Hazardous", "darkred"

        # Using columns to display results nicely
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Predicted AQI", f"{prediction:.2f}")
        
        with res_col2:
            st.info(f"**Air Quality Status:** {status}")
            st.progress(min(prediction/500, 1.0))
