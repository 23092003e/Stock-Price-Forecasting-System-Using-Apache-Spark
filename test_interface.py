import os, sys
import logging

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

train_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\train.csv"  
test_path = r"C:\Users\ADMIN\Desktop\Stock-Price-Forecasting-System-Using-Apache-Spark\data\processed\test.csv"

# Sử dụng đường dẫn tuyệt đối và đảm bảo định dạng Windows
data_path = os.path.abspath(r"data\raw\NFLX.csv")
print(f"Loading data from: {data_path}")
print(f"File exists: {os.path.exists(data_path)}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Check if data files exist
        if not os.path.exists(train_path):
            st.error(f"Training data file not found at {train_path}")
            return None, None
        if not os.path.exists(test_path):
            st.error(f"Test data file not found at {test_path}")
            return None, None

        # Load raw CSV into pandas with data validation
        train_pdf = pd.read_csv(train_path, parse_dates=['Date'])
        test_pdf = pd.read_csv(test_path, parse_dates=['Date'])

        # Validate required columns
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_columns if col not in train_pdf.columns]
        if missing_cols:
            st.error(f"Missing required columns in training data: {', '.join(missing_cols)}")
            return None, None

        # Check for missing values
        if train_pdf.isnull().any().any():
            st.warning("Training data contains missing values. These will be handled during processing.")
            # Basic missing value handling
            train_pdf = train_pdf.fillna(method='ffill').fillna(method='bfill')

        if test_pdf.isnull().any().any():
            st.warning("Test data contains missing values. These will be handled during processing.")
            test_pdf = test_pdf.fillna(method='ffill').fillna(method='bfill')

        # Initialize Spark with progress indicator
        logger.info("Initializing Spark session...")
        with st.spinner('Initializing Spark session...'):
            spark = SparkSession.builder \
                .appName("Netflix Stock Price Forecasting") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.python.worker.memory", "4g") \
                .config("spark.driver.maxResultSize", "4g") \
                .config("spark.sql.shuffle.partitions", "10") \
                .config("spark.default.parallelism", "10") \
                .config("spark.python.worker.reuse", "true") \
                .master("local[*]") \
                .getOrCreate()

        # Add enhanced statistics display
        st.sidebar.markdown("### Dataset Statistics")
        st.sidebar.write("Training Data Points:", len(train_pdf))
        st.sidebar.write("Testing Data Points:", len(test_pdf))
        st.sidebar.write("Date Range:", f"{train_pdf['Date'].min().strftime('%Y-%m-%d')} to {test_pdf['Date'].max().strftime('%Y-%m-%d')}")
        
        # Add data quality metrics
        st.sidebar.markdown("### Data Quality")
        st.sidebar.write("Training Data Completeness:", f"{(1 - train_pdf.isnull().any().sum() / len(train_pdf.columns)):.1%}")
        st.sidebar.write("Test Data Completeness:", f"{(1 - test_pdf.isnull().any().sum() / len(test_pdf.columns)):.1%}")

        # Show recent price trends with enhanced visualization
        st.markdown("### Recent Price Trends")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_pdf['Date'].tail(30), test_pdf['Close'].tail(30), label='Actual Price', color='#0077b6')
        ax.fill_between(test_pdf['Date'].tail(30), 
                       test_pdf['Low'].tail(30), 
                       test_pdf['High'].tail(30), 
                       alpha=0.2, 
                       color='#0077b6',
                       label='Price Range')
        ax.set_title('Last 30 Days Price Movement', fontsize=14, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
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
        logger.info("Loading data...")
        print(f"Python version: {sys.version}")
        print(f"Spark version: {spark.version}")
        return df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    finally:
        # Clean up Spark session
        if 'spark' in locals():
            spark.stop()

# Make predictions
def predict_lr(model, scaler, last_values, days_ahead):
    try:
        if len(last_values) < 12:
            raise ValueError("Not enough historical data for prediction (need at least 12 points)")
        
        predictions = []
        last_values = last_values.copy()
        
        for _ in range(days_ahead):
            # Reshape the data for prediction
            X = np.array(last_values[-12:]).reshape(1, -1)
            
            # Make prediction and inverse transform
            scaled_pred = model.predict(X)[0]
            pred = scaler.inverse_transform([[scaled_pred]])[0][0]
            
            # Validate prediction
            if not np.isfinite(pred):
                raise ValueError("Invalid prediction value detected")
            
            # Add prediction to the list and update last_values for next prediction
            predictions.append(pred)
            last_values.append(scaled_pred)
            last_values.pop(0)
        
        return predictions
    except Exception as e:
        st.error(f"Error in Linear Regression prediction: {str(e)}")
        return None

def predict_lstm(model, scaler, last_values, days_ahead):
    try:
        if len(last_values) < 30:
            raise ValueError("Not enough historical data for prediction (need at least 30 points)")
        
        model.eval()
        predictions = []
        
        # Create a copy of the input sequence to avoid modifying the original
        input_seq = last_values.copy()
        
        for _ in range(days_ahead):
            # Convert to PyTorch tensor and ensure correct shape
            seq = torch.FloatTensor(input_seq[-30:]).reshape(1, 30, 1)
            
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
            
            # Validate prediction
            if not np.isfinite(pred):
                raise ValueError("Invalid prediction value detected")
            
            # Add prediction to results and update input sequence
            predictions.append(pred)
            input_seq.append(scaled_pred)
        
        return predictions
    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")
        return None

def predict_hybrid(lr_model, lstm_model, hybrid_model, scaler, last_values, days_ahead):
    try:
        # Get predictions from both models
        lr_predictions = predict_lr(lr_model, scaler, last_values, days_ahead)
        lstm_predictions = predict_lstm(lstm_model, scaler, last_values, days_ahead)
        
        if lr_predictions is None or lstm_predictions is None:
            return None
        
        # Combine predictions using the hybrid model
        hybrid_predictions = hybrid_model.predict(np.column_stack((lr_predictions, lstm_predictions)))
        
        # Validate predictions
        if not np.all(np.isfinite(hybrid_predictions)):
            raise ValueError("Invalid prediction values detected in hybrid model")
        
        return hybrid_predictions
    except Exception as e:
        st.error(f"Error in Hybrid prediction: {str(e)}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="Netflix Stock Price Forecasting 🎞️",
        page_icon="📈",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .title-container {
            background-color: #E50914;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .subtitle {
            color: #333;
            text-align: center;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<div class="title-container"><h1 style="color: white;">Netflix Stock Price Forecasting 🎞️</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict Netflix stock prices using advanced machine learning models</div>', unsafe_allow_html=True)

    # Load models and data with progress indicators
    with st.spinner('Loading models...'):
        scaler, lr_model, lstm_model, hybrid_model = load_models()
    
    with st.spinner('Loading and processing data...'):
        df, test_df = load_data()

    if df is not None and test_df is not None and all([scaler, lr_model, lstm_model, hybrid_model]):
        # Add model selection in sidebar with descriptions
        st.sidebar.markdown("### Model Selection")
        model_descriptions = {
            "Linear Regression": "Traditional statistical model for linear relationships",
            "LSTM": "Deep learning model specialized for time series",
            "Hybrid Model": "Combined approach using both LR and LSTM"
        }
        selected_model = st.sidebar.selectbox(
            "Choose Prediction Model",
            list(model_descriptions.keys())
        )
        st.sidebar.markdown(f"*{model_descriptions[selected_model]}*")

        # Add prediction period selection with guidance
        st.sidebar.markdown("### Prediction Settings")
        days_ahead = st.sidebar.slider(
            "Number of Days to Predict",
            min_value=1,
            max_value=30,
            value=7,
            help="Select the number of days to forecast into the future"
        )

        # Add confidence level selection
        confidence_level = st.sidebar.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            help="Select the confidence level for prediction intervals"
        )

        # Add prediction button with loading state
        if st.sidebar.button("Generate Prediction"):
            with st.spinner(f'Generating {days_ahead}-day forecast using {selected_model}...'):
                # Get last values for prediction
                last_values = df['features'].iloc[-30:].tolist()

                # Make predictions based on selected model
                predictions = None
                if selected_model == "Linear Regression":
                    predictions = predict_lr(lr_model, scaler, last_values, days_ahead)
                elif selected_model == "LSTM":
                    predictions = predict_lstm(lstm_model, scaler, last_values, days_ahead)
                else:
                    predictions = predict_hybrid(lr_model, lstm_model, hybrid_model, scaler, last_values, days_ahead)

                if predictions is not None:
                    # Create future dates
                    last_date = test_df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=x) for x in range(1, days_ahead + 1)]

                    # Plot predictions with enhanced visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data with price range
                    historical_dates = test_df['Date'].tail(30)
                    ax.plot(historical_dates, test_df['Close'].tail(30), 
                           label='Historical Price', color='#0077b6')
                    ax.fill_between(historical_dates,
                                  test_df['Low'].tail(30),
                                  test_df['High'].tail(30),
                                  alpha=0.2,
                                  color='#0077b6',
                                  label='Historical Price Range')
                    
                    # Plot predictions with confidence interval
                    ax.plot(future_dates, predictions, 
                           label=f'{selected_model} Predictions',
                           color='#e63946',
                           linestyle='--')
                    
                    # Add confidence intervals
                    std_dev = np.std(test_df['Close'].tail(30))
                    z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
                    margin = z_score * std_dev
                    
                    ax.fill_between(future_dates,
                                  np.array(predictions) - margin,
                                  np.array(predictions) + margin,
                                  alpha=0.2,
                                  color='#e63946',
                                  label=f'{confidence_level}% Confidence Interval')
                    
                    ax.set_title(f'Netflix Stock Price Prediction ({selected_model})',
                               fontsize=14, pad=20)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price (USD)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Display prediction metrics
                    st.markdown("### Prediction Details")
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': predictions,
                        'Lower Bound': np.array(predictions) - margin,
                        'Upper Bound': np.array(predictions) + margin
                    })
                    
                    # Format the dataframe
                    for col in ['Predicted Price', 'Lower Bound', 'Upper Bound']:
                        pred_df[col] = pred_df[col].round(2)
                    
                    st.dataframe(pred_df.style.format({
                        'Date': lambda x: x.strftime('%Y-%m-%d'),
                        'Predicted Price': '${:,.2f}'.format,
                        'Lower Bound': '${:,.2f}'.format,
                        'Upper Bound': '${:,.2f}'.format
                    }))

                    # Add summary metrics
                    st.markdown("### Forecast Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Average Predicted Price", 
                                f"${np.mean(predictions):.2f}",
                                f"{((predictions[-1] - predictions[0])/predictions[0]*100):.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Minimum Predicted Price", 
                                f"${np.min(predictions):.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Maximum Predicted Price", 
                                f"${np.max(predictions):.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Failed to load required models or data. Please check the error messages above.")

if __name__ == "__main__":
    main()

# Footer with enhanced information
st.markdown("""
---
### About Netflix Stock Price Forecasting

This application uses historical stock price data for Netflix (NFLX) to predict future stock prices using three different models:

1. **Linear Regression**: A traditional statistical model that predicts future values based on linear relationships between variables.
2. **LSTM (Long Short-Term Memory)**: A deep learning model especially effective for time series forecasting.
3. **Hybrid Model**: A sophisticated approach that combines the strengths of both Linear Regression and LSTM models.

#### Features:
- Historical price trend visualization
- Multiple prediction models
- Confidence intervals for predictions
- Detailed prediction metrics
- Interactive date range selection
- Data quality indicators

**Disclaimer**: This application is for educational purposes only. Stock price predictions are not guaranteed and should not be used as financial advice.

#### Data Sources:
- Historical stock data from Yahoo Finance
- Technical indicators calculated using standard financial formulas
- Market sentiment analysis from various financial news sources
""")