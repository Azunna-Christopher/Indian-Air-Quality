import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px 

# -------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------
st.set_page_config(
    page_title="EcoVis | Air Quality Forecaster",
    page_icon="🌍",
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
# 5. DEEP DIVE ANALYTICS (INTERACTIVE & COMPARATIVE)
# -------------------------------------------
elif page == "Deep Dive Analytics":
    st.title("📊 Deep Dive Analytics")
    
    # We update the dropdown to match your specific 5 requests
    chart_type = st.selectbox(
        "Select Analysis Level:",
        [
            "1. National Trend (2015-2020)",
            "2. AQI per City per Year (Comparison)",
            "3. Pollutant Concentration Heatmap (Cities vs Chemicals)",
            "4. Top 10 Cities by Pollutant",
            "5. Pollutant Distribution Analysis"
        ]
    )
    
    st.markdown("---")

    # ---------------------------------------------------------
    # CHART 1: LINE CHART OF AVERAGE AQI IN INDIA (2015-2020)
    # ---------------------------------------------------------
    if chart_type == "1. National Trend (2015-2020)":
        st.subheader("📈 National Average AQI Trend (2015-2020)")
        st.caption("This chart aggregates data from all stations across India to show the overall trend.")
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            # Group by Date to get daily national average
            national_trend = df.groupby('Date')['AQI'].mean().reset_index()
            
            # Interactive Line Chart
            fig = px.line(
                national_trend, 
                x='Date', 
                y='AQI',
                title="Daily National Average AQI",
                color_discrete_sequence=['#2E86C1']
            )
            
            # Add a trendline (rolling average) to make it smoother
            fig.update_traces(mode="lines", opacity=0.8)
            fig.update_layout(xaxis_title="Date", yaxis_title="Average AQI")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Date column missing. Cannot plot time series.")

    # ---------------------------------------------------------
    # CHART 2: BAR CHART OF AGGREGATE AVG AQI PER CITY PER YEAR
    # ---------------------------------------------------------
    elif chart_type == "2. AQI per City per Year (Comparison)":
        st.subheader("🏙️ Aggregate Average AQI per City (Yearly Breakdown)")
        st.caption("Compare how different cities have performed across different years.")
        
        if 'Year_bin' in df.columns:
            # Group by City and Year
            city_year_aqi = df.groupby(['City', 'Year_bin'])['AQI'].mean().reset_index()
            
            # Interactive Bar Chart (Grouped)
            fig = px.bar(
                city_year_aqi, 
                x='City', 
                y='AQI', 
                color='Year_bin', 
                barmode='group',
                title="Average AQI by City and Year",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Year column missing.")

    # ---------------------------------------------------------
    # CHART 3: CORRELATION HEATMAP (CITIES vs POLLUTANTS)
    # ---------------------------------------------------------
    elif chart_type == "3. Pollutant Concentration Heatmap (Cities vs Chemicals)":
        st.subheader("🔥 Intensity Heatmap: Cities vs Pollutants")
        st.caption("Which cities have the highest intensity of specific chemical pollutants?")
        
        pollutants = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'Benzene']
        
        # Pivot table: Index=City, Columns=Pollutants, Values=Average Level
        # We use the global 'df' to show all cities
        heatmap_data = df.groupby('City')[pollutants].mean()
        
        # Interactive Heatmap
        fig = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale='RdBu_r', # Red = High Pollution
            origin='lower',
            title="Average Pollutant Concentration Matrix"
        )
        fig.update_layout(xaxis_title="Pollutant", yaxis_title="City")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # CHART 4: TOP 10 CITIES (FILTERABLE BY YEAR & POLLUTANT)
    # ---------------------------------------------------------
    elif chart_type == "4. Top 10 Cities by Pollutant":
        st.subheader("🏆 Top 10 Most Polluted Cities")
        
        c1, c2 = st.columns(2)
        with c1:
            # Filter by Year
            year_list = sorted(df['Year_bin'].unique())
            selected_year = st.selectbox("Select Year:", year_list)
        with c2:
            # Filter by Pollutant
            pollutant_list = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
            selected_pollutant = st.selectbox("Select Pollutant to Rank:", pollutant_list)
            
        # Data Processing
        ranked_df = df[df['Year_bin'] == selected_year]
        ranked_df = ranked_df.groupby('City')[selected_pollutant].mean().reset_index()
        ranked_df = ranked_df.sort_values(by=selected_pollutant, ascending=False).head(10)
        
        # Bar Chart
        fig = px.bar(
            ranked_df,
            x=selected_pollutant,
            y='City',
            orientation='h', # Horizontal for easy reading
            color=selected_pollutant,
            color_continuous_scale='Reds',
            title=f"Top 10 Cities for {selected_pollutant} in {selected_year}"
        )
        # Reverse y-axis so #1 is at top
        fig.update_layout(yaxis=dict(autorange="reversed")) 
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # CHART 5: HISTOGRAM OF DIFFERENT POLLUTANT DISTRIBUTIONS
    # ---------------------------------------------------------
    elif chart_type == "5. Pollutant Distribution Analysis":
        st.subheader("📊 Distribution of Pollutant Levels")
        st.caption("Analyze the spread and frequency of different pollutants.")
        
        # Dropdown to select specific pollutant
        pollutant_dist_list = ['AQI', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
        selected_dist_pollutant = st.selectbox("Select Pollutant:", pollutant_dist_list)
        
        # Use filtered_df (from sidebar) or df? 
        # Usually distribution is better on the sidebar selection (e.g., just show Delhi's distribution)
        # But we can clarify this in the UI
        st.info(f"Showing distribution for: **{location_title}** (Change in Sidebar)")
        
        fig = px.histogram(
            filtered_df, # Uses the Sidebar Filter!
            x=selected_dist_pollutant,
            nbins=50,
            color_discrete_sequence=['#8E44AD'],
            marginal="box", # Shows outliers
            title=f"Distribution of {selected_dist_pollutant} in {location_title}"
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

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

