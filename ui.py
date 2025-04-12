import streamlit as st
import pandas as pd
import numpy as np
import joblib

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm import SequentialLSTM


from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator

# Paths to models and data
scaler_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\scaler.pkl"
lr_model_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\linear_regressor_sklearn.pkl"
lstm_model_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\lstm_model.pth"
hybrid_model_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\hybrid_model.pkl"

train_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\train.csv"
test_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\test.csv"

scaler = joblib.load(scaler_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_spark():
    """
    Initialize Spark session.
    """
    try:
        sc = SparkContext.getOrCreate()
        spark = SparkSession.builder \
            .appName("Netflix Stock Price Forecasting") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "16g") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Error initializing Spark: {e}")
        return None
    
def load_data():
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def plot_data(data):
    try:
        fig, ax = plt.subplots()
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        ax.set_title("Stock Closing Prices Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")
    
def load_lstm_model(model_path, input_size, output_size):
    model = SequentialLSTM(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_model(chosen_model):
    try:
        if chosen_model == 'Linear Regression':
            model = joblib.load(lr_model_path)
        elif chosen_model == 'LSTM':
            model = load_lstm_model(lstm_model_path, input_size=1, output_size=1)
        elif chosen_model == 'Hybrid':
            model = joblib.load(hybrid_model_path)
        else:
            st.error("Invalid model choice")
            return None
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None