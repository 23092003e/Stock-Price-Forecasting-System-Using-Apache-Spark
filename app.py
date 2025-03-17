import streamlit as st
import pandas as pd
from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressionModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import plotly.graph_objects as go
import torch
from lstm import SequentialLSTM
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

scaler_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\scaler.pkl"
data_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\data.csv"
lr_model_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\linear_regressor_sklearn.pkl"
lstm_model_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\model\lstm_model.pth"
test_data = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\test.csv"


scaler = joblib.load(scaler_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_spark():
    try:
        spark = SparkSession.builder.appName("Netflix Stock Price Forecasting") \
        .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Failted to build Spark. Error: {e}")
        return None

def load_data():
    data = pd.read_csv(data_path)
    return data

def plot_data(data):
    chart = px.line(data, x = 'Data', y = 'Close', title="Netflix Stock Closing Prices Over Time")
    st.plotly_chart(chart, use_container_width=True)
    
# Load the model
def load_model(lr_model_path, lstm_model_path):
    lr_model = joblib.load(lr_model_path)
    lstm_model = torch.load(lstm_model_path, map_location=device)
    
    
#