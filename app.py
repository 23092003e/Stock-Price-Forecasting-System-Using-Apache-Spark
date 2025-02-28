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

scaler = joblib.load('scaler.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_spark():
    try:
        sc = SparkContext.getOrCreate()
        spark = SparkSession.builder \
            .appName("Stock Price Prediction") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark. Error: {e}")
        return None

def load_data():
    try:
        data = pd.read_csv('./data/NFLX.csv')
        return data        
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        return None

def plot_data(data):
    try:
        chart = px.line(data, x='Date', y='Close', title="Stock Closing Prices Over Time")
        st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")

def load_lstm_model(model_path, input_size, output_size):
    model = SequentialLSTM(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_model(chosen_model):
    try:
        if chosen_model == 'Linear Regressor':
            path = './models/linear_regressor'
            model = LinearRegressionModel.load(path)
        elif chosen_model == 'Decision Tree Regressor':
            path = './models/decision_tree_regressor'
            model = DecisionTreeRegressionModel.load(path)
        elif chosen_model == 'LSTM':
            path = './models/lstm.pth'
            model = load_lstm_model(path, input_size=1, output_size=1)
        else:
            st.error("Invalid model selected.")
            return None
       # st.sidebar.success(f"Model successfully loaded from {path}")
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load model. Error: {e}")
        return None

def preprocess_lstm_data(look_back, num_days, scaler = scaler):
    try:
        data = pd.read_csv('./data/test.csv').sort_values(by="Date")
        scaled_data = scaler.transform(data[['Close']].values.astype('float32'))

        def create_dataset(dataset, look_back, num_days):
            dataX = []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                if len(dataX) == num_days:  
                    break
            return np.array(dataX)

        dataX = create_dataset(scaled_data, look_back, num_days)
        dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
        
        truncated_dates = data['Date'][:len(dataX)].reset_index(drop=True)
        truncated_actuals = data['Close'][:len(dataX)].reset_index(drop=True)

        return torch.tensor(dataX, dtype=torch.float32).to(device), truncated_dates, truncated_actuals
    except Exception as e:
        st.error(f"Failed to preprocess data for LSTM. Error: {e}")
        return None, None, None


def load_data_for_prediction(spark, option, num_days=None, look_back=1, scaler=scaler):
    try:
        data_test = spark.read.csv('./data/test.csv', header=True, inferSchema=True)
        dataX, dates, actuals = preprocess_lstm_data(look_back, num_days, scaler=scaler)

        data_test = data_test.withColumn("Date", col("Date").cast("date"))
        feature_columns = ["Open", "High", "Low", "Volume"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data_test = assembler.transform(data_test).orderBy(col("Date").asc())
                
        if option == '7 days':
            data_test = data_test.limit(7)
            dataX = dataX[:7]
        elif option == '14 days':
            data_test = data_test.limit(14)
            dataX = dataX[:14]
        elif option == '30 days':
            data_test = data_test.limit(30)
            dataX = dataX[:30]
        elif option == 'Custom' and num_days:
            data_test = data_test.limit(num_days)
            dataX = dataX[:num_days]
            dates = dates[:num_days]
            actuals = actuals[:num_days]
       
        return data_test, dataX, dates, actuals
    except Exception as e:
        st.error(f"Failed to load data for prediction. Error: {e}")
        return None

def predict(spark, option, chosen_model, num_days=None, scaler = scaler, look_back=1):
    try:
        data_test,dataX, dates, actuals  = load_data_for_prediction(spark, option, num_days)
        if chosen_model == 'Linear Regressor' or chosen_model == 'Decision Tree Regressor':
            model = load_model(chosen_model)

            predictions = model.transform(data_test)
            vector_to_array_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
            predictions = predictions.withColumn("features_array", vector_to_array_udf(col("features")))
            predictions = predictions.withColumnRenamed("Close", "label")

            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
            mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
            rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
            mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
            r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"**R-squared (R2):** {r2:.4f}")

            predictions_df = predictions.select("Date", "features_array", "prediction", "label").toPandas()
            predictions_df = predictions_df.sort_values(by="Date")
            predictions_df = predictions_df.drop(columns="features_array")
            st.write(predictions_df)
            return predictions_df
        else:
            model = load_model(chosen_model)
            with torch.no_grad():
                predictions = model(dataX).detach().cpu().numpy()
            predictions = scaler.inverse_transform(predictions)
            
            predictions_df = pd.DataFrame({
                "Date": dates[:len(dataX)].reset_index(drop=True).sort_values(),
                "prediction": predictions.flatten(),
                "label": actuals[:len(dataX)].reset_index(drop=True),
            })
            predictions_df = predictions_df.sort_values(by="Date")
            mse = mean_squared_error(predictions_df['label'], predictions_df['prediction'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(predictions_df['label'], predictions_df['prediction'])
            r2 = r2_score(predictions_df['label'], predictions_df['prediction'])

            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"**R-squared (R2):** {r2:.4f}")
            st.write(predictions_df)
            return predictions_df
    except Exception as e:
        st.error(f"Failed to predict. Error: {e}")
        return None

def plot_predictions_overview(dataframe, num_days):
    try:
        # Load the original test dataset
        test = pd.read_csv('./data/test.csv')

        # Ensure the Date column is in the same format for merging
        test["Date"] = pd.to_datetime(test["Date"])
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])

        # Filter the test data to include only dates present in the predictions
        filtered_test = test[test["Date"].isin(dataframe["Date"])]

        # Merge predictions with the filtered test data
        merged_data = pd.merge(filtered_test, dataframe, on="Date", how="inner")

        # Plot actual vs predicted values using Plotly
        fig = go.Figure()

        # Add actual values (Close prices) to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["Close"],  # Original Close column from test.csv
            mode='lines',
            name='Actual Close Price',
            line=dict(color='green', width=2)
        ))

        # Add predicted values to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["prediction"],  # Predicted Close prices
            mode='lines',
            name='Predicted Close Price',
            line=dict(color='red', width=2, dash='dot')
        ))

        if num_days == 'Custom':
        # Update layout
            fig.update_layout(
                title=f"Actual vs Predicted Stock Closing Prices for the next {num_days}",
                xaxis_title="Date",
                yaxis_title="Closing Price",
                legend=dict(x=0, y=1),
                template="plotly_white"
            )
        else:
            fig.update_layout(
            title=f"Actual vs Predicted Stock Closing Prices for the next {num_days}",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")

def plot_predictions(dataframe, num_days):
    try:
        # Load the original test dataset
        test = pd.read_csv('./data/test.csv')

        # Ensure the Date column is in the same format for merging
        test["Date"] = pd.to_datetime(test["Date"])
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])

        # Merge predictions with the test data
        merged_data = pd.merge(test, dataframe, on="Date", how="left")

        # Plot actual vs predicted values using Plotly
        fig = go.Figure()

        # Add actual values (Close prices) to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["Close"],  # Original Close column from test.csv
            mode='lines',
            name='Actual Close Price',
            line=dict(color='green', width=2)
        ))

        # Add predicted values to the plot
        fig.add_trace(go.Scatter(
            x=merged_data["Date"],
            y=merged_data["prediction"],  # Predicted Close prices
            mode='lines',
            name='Predicted Close Price',
            line=dict(color='red', width=2, dash='dot')
        ))

        # Update layout
        fig.update_layout(
            title=f"Actual vs Predicted Stock Closing Prices for the next {num_days}",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot data. Error: {e}")

def main():
    st.markdown("<h1 style='text-align: center;'>Stock Price Predictions</h1>", unsafe_allow_html=True)
    spark = initialize_spark()
    if not spark:
        return

    data = load_data()
    if data is not None:
        plot_data(data)

    st.sidebar.title('User Input Features')
    chosen_model = st.sidebar.selectbox('Select Algorithm', ('Linear Regressor', 'Decision Tree Regressor','LSTM'))
    option = st.sidebar.radio('Choose a forecast period:', ['7 days', '14 days', '30 days', 'Custom'])

    num_days = None
    if option == 'Custom':
        num_days = st.sidebar.number_input('Number of days:', min_value=1, value=5)


    if st.sidebar.button('Predict'):
        prediction_df = predict(spark, option, chosen_model, num_days, scaler=scaler, look_back=1)
        if prediction_df is not None:
            # Create tabs for the two plots
            tab1, tab2 = st.tabs(["Detailed Plot", "Overview Plot"])

            with tab1:
                st.markdown("### Detailed Plot")
                plot_predictions_overview(prediction_df, num_days or option)

            with tab2:
                st.markdown("### Overview Plot")
                plot_predictions(prediction_df, num_days or option)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()