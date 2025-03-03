# Netflix Stock Price Forecasting System Using Apache Spark

This project implements a stock price forecasting system for Netflix using Apache Spark. The system analyzes historical stock market data, builds predictive models, and deploys a real-time forecasting pipeline.
## Overview

The Netflix Stock Price Prediction System uses advanced machine learning techniques to predict future stock prices, enabling informed investment decisions. The system is built with Apache Spark for scalable data processing and includes a real-time streaming component for ongoing predictions.

## Features

- **Data preprocessing and cleaning pipeline** for stock market data
- **Technical indicator generation** including moving averages, RSI, and Bollinger Bands
- **Multiple prediction models** (Linear Regression, AMIRA, Long Short Term Memory)
- **Real-time prediction streaming** for continuous forecasting
- **Interactive dashboard**** for visualizing predictions and model performance
- **Alerting system** for significant price movement detection

## Technologies & Tools
- **Knowledge**: Machine Learning, Deep Learning, Big Data Technology
- **Apache Spark:** Distributed data processing and model training.
- **Python:** Main programming language.
- **Jupyter Notebook:** Primary development environment (see `src.ipynb`).
- **Additional Libraries:** Refer to `requirement.txt` for the complete list of required packages.


## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/23092003e/Stock-Price-Forecasting-System-Using-Apache-Spark.git
   cd Stock-Price-Forecasting-System-Using-Apache-Spark


2. **Install the required libraries: Use the following command to install the dependencies:**
    ```bash
    pip install -r requirement.txt

3. **Run the Project:**
    - Open src.ipynb using Jupyter Notebook.
    - Execute the notebook cells in order to preprocess the data, train the models, and generate stock price forecasts.

4. **Results & Evaluation**
   Upon running the project, you will obtain:
    - Real-time stock price forecasts.
    - Visualizations and performance metrics that evaluate the model's accuracy.
    - Insights derived from the analysis of historical stock market data.

## Project Structure
#### Stock price Forecasting System using Apache Spark/

├── data/                  # Raw and processed data
├── image/                
├── models/
├── notebooks/               
├── helper_function.py
├── lstm.py
├── src.py
├── dashboard.py    
├── requirement.txt
└── README.md

## Performance Optimization
**The system includes several Spark optimizations:**

- Memory allocation tuning for executors and drivers
- Parallelism configuration for efficient resource utilization
- Data compression settings to minimize I/O overhead
- Caching strategies for frequently accessed data

## Regulatory Compliance
This system is designed as an analytical tool and does not provide financial advice. Users should consult financial professionals before making investment decisions based on predictions.

### License
This project is provided under the MIT License. See LICENSE file for details.

### Contributors

Manh Hoang (23092003e)

### Acknowledgments

- Financial data provided by Kaggle
- Built with Apache Spark

##### If you have any questions, Just contact me: hoangvanmanh2309@gmail.com

    
