from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, to_timestamp
import numpy as np
# Help function to set model parameters
def set_model_params(model, param_map):
    for param, value in param_map.items():
        model.set(param, value)
    return model

# Model training function
def train_model(train_set, test_set, model, evaluator, paramGrid,
                metrics = ['rmse', 'mae','r2']):

    min_score = float('inf')
    best_model = None

    # Grid search over parameters
    for params in paramGrid:
        # Set Parameters
        model = set_model_params(model, params)

        # Train the model
        trained_model = model.fit(train_set)

        # Evaluate on trainset
        preds_train = trained_model.transform(train_set)
        score = evaluator.evaluate(preds_train)

        # Update best model if score improves
        if score < min_score:
            best_model = trained_model
            min_score = score

    preds_test = best_model.transform(test_set)

    # Add prediction column to the test DataFrame
    predictions = test_set.join(preds_test.select("Date","prediction"), on="Date", how="left")

    # Calculate evaluation metrics
    train_metrics = {}
    test_metrics = {}

    for m in metrics:
        evaluator.setMetricName(m)
        test_metrics[m] = evaluator.evaluate(preds_test)
        train_metrics[m] = evaluator.evaluate(preds_train)

    # Print dataframe
    print("DataFrame: ")
    print(predictions.show(5, truncate=False))

    # Print evaluation metrics
    print("Train Metrics: ", train_metrics)
    print("Test Metrics: ", test_metrics)  

    return best_model, predictions, train_metrics, test_metrics 




#### LSTM Helper Function
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def inverse_scale(scaled_vector, data):
    X_max = data.select('close').rdd.max()[0]
    X_min = data.select('close').rdd.min()[0]
    scaler_min = 0.0
    scaler_max = 1.0
    return (scaler_max * X_min - scaler_min * X_max - X_min * scaled_vector + X_max * scaled_vector) / \
                  (scaler_max - scaler_min)