import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import findspark
findspark.init()

# Configure Spark environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Configure Hadoop for Windows
hadoop_home = os.path.abspath(os.path.join(os.getcwd(), "hadoop")).replace("\\", "/")
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + os.path.join(hadoop_home, "bin").replace("\\", "/")
os.environ["HADOOP_CONF_DIR"] = os.path.join(hadoop_home, "etc", "hadoop").replace("\\", "/")

from pyspark.sql import SparkSession

# Set page config
st.set_page_config(
    page_title="Netflix Stock Price History",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .title {
        color: #E50914;
        text-align: center;
        padding: 20px;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    try:
        # Create Spark session with updated configuration
        spark = SparkSession.builder \
            .appName("Netflix Stock Price History") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={hadoop_home}") \
            .config("spark.hadoop.hadoop.home.dir", hadoop_home) \
            .master("local[*]") \
            .getOrCreate()

        # Load data
        data_path = os.path.abspath("data/raw/NFLX.csv")
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            return None

        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Convert to pandas for easier visualization
        pdf = df.toPandas()
        # Ensure Date column is datetime
        pdf['Date'] = pd.to_datetime(pdf['Date'])
        return pdf
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    # Title
    st.markdown("<h1 class='title'>Netflix Stock Price History 📈</h1>", unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Sidebar
        st.sidebar.header("Controls")
        
        # Date range selector
        st.sidebar.subheader("Select Date Range")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data based on date range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        
        if len(filtered_df) == 0:
            st.warning("No data available for the selected date range.")
            return
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stock Price Overview")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(filtered_df['Date'], filtered_df['Close'], label='Close Price')
            ax.set_title('Netflix Stock Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Trading Volume")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(filtered_df['Date'], filtered_df['Volume'], alpha=0.7)
            ax.set_title('Trading Volume')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Statistics
        st.subheader("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Price", f"${filtered_df['Close'].mean():.2f}")
        with col2:
            st.metric("Highest Price", f"${filtered_df['High'].max():.2f}")
        with col3:
            st.metric("Lowest Price", f"${filtered_df['Low'].min():.2f}")
        with col4:
            price_change = ((filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]) / filtered_df['Close'].iloc[0]) * 100
            st.metric("Price Change", f"{price_change:.1f}%")
        
        # Data Table
        st.subheader("Historical Data")
        display_df = filtered_df.copy()
        display_df['Date'] = display_df['Date'].dt.date  # Convert to date for display
        st.dataframe(display_df.style.format({
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Low': '${:.2f}',
            'Close': '${:.2f}',
            'Adj Close': '${:.2f}',
            'Volume': '{:,.0f}'
        }))
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="netflix_stock_data.csv",
            mime="text/csv"
        )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        ### About this Dashboard
        This dashboard provides historical visualization of Netflix stock price data. Features include:
        - Interactive date range selection
        - Price and volume charts
        - Key statistics
        - Downloadable historical data
        
        Data source: Yahoo Finance
        """)

if __name__ == "__main__":
    main() 