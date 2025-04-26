import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
import os
from sklearn.preprocessing import MinMaxScaler

# Paths to models and data
scaler_path = "model/scaler.pkl"
lr_model_path = "model/linear_regressor_sklearn.pkl"
lstm_model_path = "model/lstm_model.pth"
hybrid_model_path = "model/hybrid_model.pkl"

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

# Define LSTM model architecture to match the saved model
# class LSTMModel(torch.nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)
#         self.linear = torch.nn.Linear(hidden_layer_size, output_size)
#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
#                             torch.zeros(1, 1, self.hidden_layer_size))

#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]
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
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Load historical data
        df = pd.read_csv(train_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Load test data
        test_df = pd.read_csv(test_path)
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
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

# App UI
st.set_page_config(page_title="Netflix Stock Price Forecasting", layout="wide")

st.title("Netflix Stock Price Forecasting System")
st.markdown("""
This application predicts Netflix stock prices using Linear Regression and LSTM models.
""")

# Load models and data
scaler, lr_model, lstm_model, hybrid_model = load_models()
df, test_df = load_data()

if scaler is None or lr_model is None or lstm_model is None or hybrid_model is None or df is None or test_df is None:
    st.error("Failed to load required models or data. Please check the file paths.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model", 
    ["Linear Regression", "LSTM", "Hybrid", "Compare All"]
)

# Number of days to predict
days_to_predict = st.sidebar.slider(
    "Days to Predict Ahead", 
    min_value=1, 
    max_value=30, 
    value=7
)

# Date range for historical data
date_range = st.sidebar.date_input(
    "Select Date Range for Historical Data",
    value=(df['Date'].min().date(), df['Date'].max().date()),
    min_value=df['Date'].min().date(),
    max_value=df['Date'].max().date()
)

# Filter data based on date range
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
else:
    filtered_df = df

# Main content
tab1, tab2, tab3 = st.tabs(["Prediction", "Historical Data", "Performance Analysis"])

with tab1:
    st.header("Stock Price Prediction")
    
    # Get the latest data for prediction
    last_actual_date = filtered_df['Date'].max()
    
    # Check if enough data is available
    if len(filtered_df['Close']) < 30:
        st.error("Not enough data points for prediction. Please select a wider date range.")
        st.stop()
    
    # For simplicity, use the last 30 values for prediction
    last_prices = filtered_df['Close'][-30:].values
    
    # Scale the values for prediction
    scaled_prices = scaler.transform(last_prices.reshape(-1, 1)).flatten()
    
    if st.button("Generate Prediction"):
        st.subheader(f"Netflix Stock Price Prediction (Next {days_to_predict} Days)")
        
        # Generate dates for future predictions
        future_dates = [last_actual_date + timedelta(days=i+1) for i in range(days_to_predict)]
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Container for prediction results
        prediction_container = st.container()
        
        with prediction_container:
            if model_type == "Linear Regression" or model_type == "Compare All":
                # Make LR predictions
                lr_predictions = predict_lr(lr_model, scaler, list(scaled_prices), days_to_predict)
                
                if model_type == "Linear Regression":
                    # Create DataFrame for display
                    pred_df = pd.DataFrame({
                        'Date': future_dates_str,
                        'Predicted Price (USD)': lr_predictions
                    })
                    
                    st.write("Linear Regression Model Predictions:")
                    st.dataframe(pred_df)
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot historical data
                    ax.plot(filtered_df['Date'][-30:], filtered_df['Close'][-30:], label='Historical Prices', color='blue')
                    
                    # Plot predictions
                    ax.plot(future_dates, lr_predictions, label='LR Predictions', color='green', linestyle='--')
                    
                    ax.set_title('Netflix Stock Price Prediction (Linear Regression)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price (USD)')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
            if model_type == "LSTM" or model_type == "Compare All":
                # Make LSTM predictions
                lstm_predictions = predict_lstm(lstm_model, scaler, list(scaled_prices), days_to_predict)
                
                if model_type == "LSTM":
                    # Create DataFrame for display
                    pred_df = pd.DataFrame({
                        'Date': future_dates_str,
                        'Predicted Price (USD)': lstm_predictions
                    })
                    
                    st.write("LSTM Model Predictions:")
                    st.dataframe(pred_df)
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot historical data
                    ax.plot(filtered_df['Date'][-30:], filtered_df['Close'][-30:], label='Historical Prices', color='blue')
                    
                    # Plot predictions
                    ax.plot(future_dates, lstm_predictions, label='LSTM Predictions', color='red', linestyle='--')
                    
                    ax.set_title('Netflix Stock Price Prediction (LSTM)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price (USD)')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
            if model_type == "Hybrid" or model_type == "Compare All":
                # Make Hybrid predictions
                hybrid_predictions = predict_hybrid(lr_model, lstm_model, hybrid_model, scaler, list(scaled_prices), days_to_predict)
                
                if model_type == "Hybrid":
                    # Create DataFrame for display
                    pred_df = pd.DataFrame({
                        'Date': future_dates_str,
                        'Predicted Price (USD)': hybrid_predictions
                    })
                    
                    st.write("Hybrid Model Predictions:")
                    st.dataframe(pred_df)
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot historical data
                    ax.plot(filtered_df['Date'][-30:], filtered_df['Close'][-30:], label='Historical Prices', color='blue')
                    
                    # Plot predictions
                    ax.plot(future_dates, hybrid_predictions, label='Hybrid Predictions', color='purple', linestyle='--')
                    
                    ax.set_title('Netflix Stock Price Prediction (Hybrid)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price (USD)')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
            if model_type == "Compare All":
                # Create DataFrame for comparison
                comparison_df = pd.DataFrame({
                    'Date': future_dates_str,
                    'Linear Regression': lr_predictions,
                    'LSTM': lstm_predictions,
                    'Hybrid': hybrid_predictions
                })
                
                # Plot comparison
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                ax.plot(filtered_df['Date'][-30:], filtered_df['Close'][-30:], label='Historical Prices', color='blue')
                
                # Plot predictions
                ax.plot(future_dates, lr_predictions, label='Linear Regression', color='green', linestyle='--')
                ax.plot(future_dates, lstm_predictions, label='LSTM', color='red', linestyle='-.')
                ax.plot(future_dates, hybrid_predictions, label='Hybrid', color='purple', linestyle=':')
                
                ax.set_title(f'Netflix Stock Price Predictions (Next {days_to_predict} Days)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display the comparison table
                st.subheader("Prediction Comparison")
                st.dataframe(comparison_df)
                
                # Calculate and display metrics for comparison
                st.subheader("Model Performance Metrics")
                
                # Get actual values for comparison
                actual_values = filtered_df['Close'].values[-days_to_predict:]
                
                # Calculate metrics for all models
                # Linear Regression metrics
                lr_mae = np.mean(np.abs((lr_predictions - actual_values) / actual_values))
                lr_rmse = np.sqrt(np.mean(((lr_predictions - actual_values) / actual_values) ** 2))
                
                # LSTM metrics
                lstm_mae = np.mean(np.abs((lstm_predictions - actual_values) / actual_values))
                lstm_rmse = np.sqrt(np.mean(((lstm_predictions - actual_values) / actual_values) ** 2))
                
                # Hybrid Model metrics
                hybrid_mae = np.mean(np.abs((hybrid_predictions - actual_values) / actual_values))
                hybrid_rmse = np.sqrt(np.mean(((hybrid_predictions - actual_values) / actual_values) ** 2))
                
                # Display metrics in text format
                st.text("Linear Regression:")
                st.text(f"MAE: {lr_mae:.6f}, RMSE: {lr_rmse:.6f}")
                st.text("")  # Add empty line
                st.text("LSTM:")
                st.text(f"MAE: {lstm_mae:.6f}, RMSE: {lstm_rmse:.6f}")
                st.text("")  # Add empty line
                st.text("Hybrid Model:")
                st.text(f"MAE: {hybrid_mae:.6f}, RMSE: {hybrid_rmse:.6f}")
                st.text("")  # Add empty line
                
                # Calculate percentage differences
                lr_diff = np.mean(np.abs((lr_predictions - hybrid_predictions) / hybrid_predictions)) * 100
                lstm_diff = np.mean(np.abs((lstm_predictions - hybrid_predictions) / hybrid_predictions)) * 100
                
                st.text("Average Percentage Difference from Hybrid Model:")
                st.text(f"Linear Regression: {lr_diff:.2f}%")
                st.text(f"LSTM: {lstm_diff:.2f}%")
                
                # Plot error comparison
                st.subheader("Error Comparison")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot absolute errors
                ax1.plot(future_dates, np.abs(np.array(lr_predictions) - np.array(hybrid_predictions)), 
                        label='Linear Regression', color='green')
                ax1.plot(future_dates, np.abs(np.array(lstm_predictions) - np.array(hybrid_predictions)), 
                        label='LSTM', color='red')
                ax1.set_title('Absolute Error from Hybrid Model')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Absolute Error')
                ax1.legend()
                ax1.grid(True)
                plt.xticks(rotation=45)
                
                # Plot percentage errors
                ax2.plot(future_dates, np.abs((np.array(lr_predictions) - np.array(hybrid_predictions)) / np.array(hybrid_predictions)) * 100, 
                        label='Linear Regression', color='green')
                ax2.plot(future_dates, np.abs((np.array(lstm_predictions) - np.array(hybrid_predictions)) / np.array(hybrid_predictions)) * 100, 
                        label='LSTM', color='red')
                ax2.set_title('Percentage Error from Hybrid Model')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Percentage Error (%)')
                ax2.legend()
                ax2.grid(True)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)

with tab2:
    st.header("Historical Stock Data")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Closing Price", "Volume", "All OHLC Data", "Daily Returns"]
    )
    
    if chart_type == "Closing Price":
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df['Date'], filtered_df['Close'], label='Close Price', color='blue')
        ax.set_title('Netflix Historical Closing Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif chart_type == "Volume":
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(filtered_df['Date'], filtered_df['Volume'], color='green', alpha=0.7)
        ax.set_title('Netflix Trading Volume')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif chart_type == "All OHLC Data":
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df['Date'], filtered_df['Open'], label='Open', color='green')
        ax.plot(filtered_df['Date'], filtered_df['High'], label='High', color='red')
        ax.plot(filtered_df['Date'], filtered_df['Low'], label='Low', color='purple')
        ax.plot(filtered_df['Date'], filtered_df['Close'], label='Close', color='blue')
        ax.set_title('Netflix OHLC Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif chart_type == "Daily Returns":
        # Calculate daily returns
        filtered_df['Daily Return'] = filtered_df['Close'].pct_change() * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df['Date'][1:], filtered_df['Daily Return'][1:], color='blue')
        ax.set_title('Netflix Daily Returns (%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show the raw data
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)

with tab3:
    st.header("Model Performance Analysis")
    
    # Use test data to evaluate model performance
    if test_df is not None:
        # Check if test data has enough points
        if len(test_df) < 30:
            st.error("Test dataset is too small for performance analysis. Need at least 30 data points.")
            st.stop()
        
        # Prepare test data
        X_test = test_df['Close'].values[:-1]
        y_test = test_df['Close'].values[1:]
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).flatten()
        
        # Make predictions using all models
        lr_test_preds = []
        lstm_test_preds = []
        hybrid_test_preds = []
        
        # Linear Regression predictions
        for i in range(len(X_test) - 11):
            lr_input = X_test_scaled[i:i+12].reshape(1, -1)
            lr_pred = lr_model.predict(lr_input)[0]
            lr_pred_unscaled = scaler.inverse_transform([[lr_pred]])[0][0]
            lr_test_preds.append(lr_pred_unscaled)
        
        # LSTM predictions
        for i in range(len(X_test) - 29):
            lstm_input = torch.FloatTensor(X_test_scaled[i:i+30]).reshape(1, 30, 1)
            lstm_model.hidden_cell1 = (torch.zeros(1, 1, lstm_model.hidden_size1),
                                    torch.zeros(1, 1, lstm_model.hidden_size1))
            lstm_model.hidden_cell2 = (torch.zeros(1, 1, lstm_model.hidden_size2),
                                    torch.zeros(1, 1, lstm_model.hidden_size2))
            lstm_model.hidden_cell3 = (torch.zeros(1, 1, lstm_model.hidden_size3),
                                    torch.zeros(1, 1, lstm_model.hidden_size3))
            with torch.no_grad():
                lstm_pred = lstm_model(lstm_input).item()
            lstm_pred_unscaled = scaler.inverse_transform([[lstm_pred]])[0][0]
            lstm_test_preds.append(lstm_pred_unscaled)
            
            # Hybrid predictions
            if i < len(lr_test_preds):  # Only create hybrid prediction if we have both LR and LSTM
                hybrid_pred = hybrid_model.predict(np.column_stack(([lr_test_preds[i]], [lstm_pred_unscaled])))[0]
                hybrid_test_preds.append(hybrid_pred)
        
        # Get the minimum length
        min_len = min(len(lr_test_preds), len(lstm_test_preds), len(hybrid_test_preds), len(y_test)-30)
        
        # Trim all arrays to the same length
        lr_test_preds = np.array(lr_test_preds[:min_len])
        lstm_test_preds = np.array(lstm_test_preds[:min_len])
        hybrid_test_preds = np.array(hybrid_test_preds[:min_len])
        y_test_aligned = y_test[30:30+min_len]  # Start from 30 to align with LSTM window
        
        # Calculate metrics for all models
        # Linear Regression metrics
        lr_mae = np.mean(np.abs((lr_test_preds - y_test_aligned) / y_test_aligned))
        lr_rmse = np.sqrt(np.mean(((lr_test_preds - y_test_aligned) / y_test_aligned) ** 2))
        
        # LSTM metrics
        lstm_mae = np.mean(np.abs((lstm_test_preds - y_test_aligned) / y_test_aligned))
        lstm_rmse = np.sqrt(np.mean(((lstm_test_preds - y_test_aligned) / y_test_aligned) ** 2))
        
        # Hybrid Model metrics
        hybrid_mae = np.mean(np.abs((hybrid_test_preds - y_test_aligned) / y_test_aligned))
        hybrid_rmse = np.sqrt(np.mean(((hybrid_test_preds - y_test_aligned) / y_test_aligned) ** 2))
        
        # Display metrics in text format
        st.subheader("Model Performance Metrics")
        st.text("Linear Regression:")
        st.text(f"MAE: {lr_mae:.6f}, RMSE: {lr_rmse:.6f}")
        st.text("")  # Add empty line
        st.text("LSTM:")
        st.text(f"MAE: {lstm_mae:.6f}, RMSE: {lstm_rmse:.6f}")
        st.text("")  # Add empty line
        st.text("Hybrid Model:")
        st.text(f"MAE: {hybrid_mae:.6f}, RMSE: {hybrid_rmse:.6f}")
        st.text("")  # Add empty line
        
        # Calculate percentage differences
        lr_diff = np.mean(np.abs((lr_test_preds - hybrid_test_preds) / hybrid_test_preds)) * 100
        lstm_diff = np.mean(np.abs((lstm_test_preds - hybrid_test_preds) / hybrid_test_preds)) * 100
        
        st.text("Average Percentage Difference from Hybrid Model:")
        st.text(f"Linear Regression: {lr_diff:.2f}%")
        st.text(f"LSTM: {lstm_diff:.2f}%")
        
        # Plot actual vs predicted values
        st.subheader("Actual vs Predicted Values")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Linear Regression plot
        ax1.plot(test_df['Date'][30:30+min_len], y_test_aligned, label='Actual', color='blue')
        ax1.plot(test_df['Date'][30:30+min_len], lr_test_preds, label='Predicted (LR)', color='green', linestyle='--')
        ax1.set_title('Linear Regression: Actual vs Predicted Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)
        
        # LSTM plot
        ax2.plot(test_df['Date'][30:30+min_len], y_test_aligned, label='Actual', color='blue')
        ax2.plot(test_df['Date'][30:30+min_len], lstm_test_preds, label='Predicted (LSTM)', color='red', linestyle='--')
        ax2.set_title('LSTM: Actual vs Predicted Price')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True)
        
        # Hybrid plot
        ax3.plot(test_df['Date'][30:30+min_len], y_test_aligned, label='Actual', color='blue')
        ax3.plot(test_df['Date'][30:30+min_len], hybrid_test_preds, label='Predicted (Hybrid)', color='purple', linestyle='--')
        ax3.set_title('Hybrid: Actual vs Predicted Price')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price (USD)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Error distribution
        st.subheader("Error Distribution")
        
        lr_errors = lr_test_preds - y_test_aligned
        lstm_errors = lstm_test_preds - y_test_aligned
        hybrid_errors = hybrid_test_preds - y_test_aligned
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Linear Regression error distribution
        sns.histplot(lr_errors, kde=True, ax=ax1, color='green')
        ax1.set_title('Linear Regression Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        
        # LSTM error distribution
        sns.histplot(lstm_errors, kde=True, ax=ax2, color='red')
        ax2.set_title('LSTM Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
        
        # Hybrid error distribution
        sns.histplot(hybrid_errors, kde=True, ax=ax3, color='purple')
        ax3.set_title('Hybrid Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("Test data could not be loaded. Performance analysis is not available.")

# Footer
st.markdown("""
---
### About Netflix Stock Price Forecasting

This application uses historical stock price data for Netflix (NFLX) to predict future stock prices using two different models:

1. **Linear Regression**: A traditional statistical model that predicts future values based on linear relationships between variables.
2. **LSTM (Long Short-Term Memory)**: A deep learning model especially effective for time series forecasting.

**Disclaimer**: This application is for educational purposes only. Stock price predictions are not guaranteed and should not be used as financial advice.
""")