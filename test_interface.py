import os, sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os

# Spark imports for scaling
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

# Paths to models and data
scaler_path = "model/scaler.pkl"
lr_model_path = "model/linear_regressor_sklearn.pkl"
lstm_model_path = "model/lstm_model.pth"
hybrid_model_path = "model/hybrid_model.pkl"

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        
        # Define LSTM layers with correct input and hidden sizes
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size=128, batch_first=True)
        self.lstm2 = torch.nn.LSTM(128, hidden_size=64, batch_first=True)
        self.lstm3 = torch.nn.LSTM(64, hidden_size=32, batch_first=True)
        
        # Define fully connected layers
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, output_size)
        
        # Hidden sizes for reference
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.hidden_size3 = 32
        
        # Initialize hidden states (will be set externally)
        self.hidden_cell1 = None
        self.hidden_cell2 = None
        self.hidden_cell3 = None

    def forward(self, input_seq):
        # Ensure input_seq is 2D (seq_len, input_size) and add batch dimension
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(1)  # Shape: (seq_len, 1, input_size)
        elif input_seq.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {input_seq.shape}")
        
        # Pass through LSTM layers using provided hidden states
        lstm_out, self.hidden_cell1 = self.lstm1(input_seq, self.hidden_cell1)
        lstm_out, self.hidden_cell2 = self.lstm2(lstm_out, self.hidden_cell2)
        lstm_out, self.hidden_cell3 = self.lstm3(lstm_out, self.hidden_cell3)
        
        # Pass through fully connected layers (use last time step)
        out = self.fc1(lstm_out[:, -1, :])
        predictions = self.fc2(out)
        
        return predictions.squeeze()

# Load models and data
@st.cache_resource
def load_models():
    try:
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = load(f)
        
        # Load Linear Regression model
        with open(lr_model_path, 'rb') as f:
            lr_model = load(f)
        
        # Load LSTM model
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device('cpu')))
        lstm_model.eval()
        
        # Load hybrid model
        with open(hybrid_model_path, 'rb') as f:
            hybrid_model = load(f)
        
        return scaler, lr_model, lstm_model, hybrid_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

@st.cache_data
def load_data():
    try:
        # Load raw CSV into pandas
        train_pdf = pd.read_csv(train_path, parse_dates=['Date'])
        test_pdf  = pd.read_csv(test_path,  parse_dates=['Date'])

        # Initialize Spark
        spark = (SparkSession.builder
                .master("local[*]")
                .appName("StreamlitScaling")
                .config("spark.pyspark.python", sys.executable)
                .config("spark.pyspark.driver.python", sys.executable)
                .getOrCreate())

        # Add basic statistics display
        st.sidebar.markdown("### Dataset Statistics")
        st.sidebar.write("Training Data Points:", len(train_pdf))
        st.sidebar.write("Testing Data Points:", len(test_pdf))
        st.sidebar.write("Date Range:", f"{train_pdf['Date'].min().strftime('%Y-%m-%d')} to {test_pdf['Date'].max().strftime('%Y-%m-%d')}")

        # Show recent price trends
        st.markdown("### Recent Price Trends")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_pdf['Date'].tail(30), test_pdf['Close'].tail(30), label='Actual Price')
        ax.set_title('Last 30 Days Price Movement')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)

        # Define feature columns and pipeline
        feature_cols = ["Open", "High", "Low", "Volume", "Price_Range",
                        "Daily_Change", "MA_10", "MA_50", "RSI",
                        "Upper_BB", "Lower_BB", "Stoch_Osc"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        scaler_spark = MinMaxScaler(inputCol='features', outputCol='scaled_features')
        pipeline = Pipeline(stages=[assembler, scaler_spark])

        # Create Spark DataFrames
        train_sdf = spark.createDataFrame(train_pdf)
        test_sdf  = spark.createDataFrame(test_pdf)

        # Fit and transform
        transformer      = pipeline.fit(train_sdf)
        train_scaled_sdf = transformer.transform(train_sdf)
        test_scaled_sdf  = transformer.transform(test_sdf)

        # UDF to convert vector to Python list
        to_list = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
        train_scaled_sdf = train_scaled_sdf.withColumn("features", to_list("scaled_features")).select("Date","features","Close")
        test_scaled_sdf  = test_scaled_sdf.withColumn("features", to_list("scaled_features")).select("Date","features","Close")

        # Convert back to pandas
        df      = train_scaled_sdf.toPandas()
        test_df = test_scaled_sdf.toPandas()
        return df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Make predictions
def predict_lr(model, scaler, last_values, days_ahead):
    predictions = []
    last_values = last_values.copy()
    
    for _ in range(days_ahead):
        # Reshape the data for prediction
        X = np.array(last_values[-12:]).reshape(1, -1)
        
        # Make prediction and inverse transform
        scaled_pred = model.predict(X)[0]
        pred = scaler.inverse_transform([[scaled_pred]])[0][0]
        
        # Add prediction to the list and update last_values for next prediction
        predictions.append(pred)
        last_values.append(scaled_pred)
        last_values.pop(0)
    
    return predictions

def predict_lstm(model, scaler, last_values, days_ahead):
    model.eval()
    predictions = []
    
    # Create a copy of the input sequence to avoid modifying the original
    input_seq = last_values.copy()
    
    for _ in range(days_ahead):
        # Convert to PyTorch tensor and ensure correct shape
        seq = torch.FloatTensor(input_seq[-30:]).reshape(1, 30, 1)  # Shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden states for each LSTM layer
        model.hidden_cell1 = (torch.zeros(1, 1, model.hidden_size1),
                              torch.zeros(1, 1, model.hidden_size1))
        model.hidden_cell2 = (torch.zeros(1, 1, model.hidden_size2),
                              torch.zeros(1, 1, model.hidden_size2))
        model.hidden_cell3 = (torch.zeros(1, 1, model.hidden_size3),
                              torch.zeros(1, 1, model.hidden_size3))
        
        # Make prediction
        with torch.no_grad():
            scaled_pred = model(seq).item()
        
        # Inverse transform to get the actual price
        pred = scaler.inverse_transform([[scaled_pred]])[0][0]
        
        # Add prediction to results and update input sequence
        predictions.append(pred)
        input_seq.append(scaled_pred)
    
    return predictions

def predict_hybrid(lr_model, lstm_model, hybrid_model, scaler, last_values, days_ahead):
    # Get predictions from both models
    lr_predictions = predict_lr(lr_model, scaler, last_values, days_ahead)
    lstm_predictions = predict_lstm(lstm_model, scaler, last_values, days_ahead)
    
    # Combine predictions using the hybrid model
    hybrid_predictions = hybrid_model.predict(np.column_stack((lr_predictions, lstm_predictions)))
    
    return hybrid_predictions

def main():
    # Load models and data
    scaler, lr_model, lstm_model, hybrid_model = load_models()
    df, test_df = load_data()

    if df is not None and test_df is not None:
        # Add model selection in sidebar
        st.sidebar.markdown("### Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose Prediction Model",
            ["Linear Regression", "LSTM", "Hybrid Model"]
        )

        # Add prediction period selection
        days_ahead = st.sidebar.slider("Prediction Days", 1, 30, 7)

        # Add prediction button
        if st.sidebar.button("Generate Prediction"):
            st.markdown("### Price Predictions")
            
            # Get last values for prediction
            last_values = df['features'].iloc[-30:].tolist()

            # Make predictions based on selected model
            if selected_model == "Linear Regression":
                predictions = predict_lr(lr_model, scaler, last_values, days_ahead)
            elif selected_model == "LSTM":
                predictions = predict_lstm(lstm_model, scaler, last_values, days_ahead)
            else:
                predictions = predict_hybrid(lr_model, lstm_model, hybrid_model, scaler, last_values, days_ahead)

            # Create future dates
            last_date = test_df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, days_ahead + 1)]

            # Plot predictions
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(test_df['Date'].tail(30), test_df['Close'].tail(30), 
                   label='Historical Price', color='blue')
            
            # Plot predictions
            ax.plot(future_dates, predictions, 
                   label=f'{selected_model} Predictions', color='red', linestyle='--')
            
            ax.set_title(f'Netflix Stock Price Prediction ({selected_model})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Display prediction metrics
            st.markdown("### Prediction Details")
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions
            })
            st.dataframe(pred_df)

if __name__ == "__main__":
    main()

# Footer
st.markdown("""
---
### About Netflix Stock Price Forecasting

This application uses historical stock price data for Netflix (NFLX) to predict future stock prices using two different models:

1. **Linear Regression**: A traditional statistical model that predicts future values based on linear relationships between variables.
2. **LSTM (Long Short-Term Memory)**: A deep learning model especially effective for time series forecasting.

**Disclaimer**: This application is for educational purposes only. Stock price predictions are not guaranteed and should not be used as financial advice.
""")