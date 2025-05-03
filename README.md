# Netflix Stock Price Forecasting System Using Apache Spark 🚀
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/23092003e/Stock-Price-Forecasting-System-Using-Apache-Spark/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A scalable, real-time stock price forecasting system for Netflix, powered by Apache Spark and advanced machine learning.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples & Screenshots](#examples--screenshots)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Authors & Contact](#authors--contact)

---

## Project Overview

**Netflix Stock Price Forecasting System** leverages Apache Spark for distributed data processing and machine learning to predict Netflix (NFLX) stock prices. The system provides an interactive dashboard for real-time forecasting, historical analysis, and model performance evaluation.


---

## Features

- ⚡ **Distributed Data Processing** with Apache Spark
- 📈 **Technical Indicator Generation** (MA, RSI, Bollinger Bands, etc.)
- 🤖 **Multiple Prediction Models**: Linear Regression, LSTM, Hybrid
- 🖥️ **Interactive Streamlit Dashboard** for visualization and analysis
- 🔔 **Real-time Prediction Streaming** and alerting for significant price changes
- 📊 **Performance Metrics** and error analysis
- 🛡️ **Data Quality Checks** and preprocessing pipeline

---

## Architecture & Tech Stack

**Architecture Overview:**

```
Raw Data (CSV) → Data Preprocessing (Spark) → Feature Engineering → Model Training (LR, LSTM, Hybrid) → 
Model Serialization → Streamlit Dashboard (Real-time Prediction & Visualization)
```

**Main Technologies:**

- **Python 3.8+**
- **Apache Spark** (PySpark)
- **Pandas, NumPy, Scikit-learn**
- **PyTorch** (for LSTM)
- **Streamlit** (dashboard)
- **Matplotlib, Seaborn** (visualization)
- **Joblib** (model serialization)

---

## Installation

### Requirements

- **OS:** Linux, macOS, or Windows
- **Python:** 3.8 or higher
- **Java:** 8+ (for Apache Spark)
- **pip** (Python package manager)

### Clone & Install Dependencies

```bash
git clone https://github.com/23092003e/Stock-Price-Forecasting-System-Using-Apache-Spark.git
cd Stock-Price-Forecasting-System-Using-Apache-Spark
pip install -r requirement.txt
```

---

## Configuration

- **Data:** Place your raw and processed CSV files in the `data/` directory.
- **Model Paths:** Update model and scaler paths in `app.py` or `test_interface.py` if needed.
- **Environment Variables:**  
  For PySpark compatibility, set:
  ```bash
  export PYSPARK_PYTHON=$(which python)
  export PYSPARK_DRIVER_PYTHON=$(which python)
  ```
- **Config File:** (Optional) Create a `.env` or `config.yaml` for custom settings.

---

## Usage

### Run the Dashboard

```bash
streamlit run app.py
```
or
```bash
streamlit run test_interface.py
```

### Options & Environment Variables

- `--server.port`: Set custom port for Streamlit (default: 8501)
- `--server.headless true`: Run in headless mode (for servers)
- **Model/Data Paths:** Edit in the script or via environment variables

---

## Examples & Screenshots

**Example: Data Scaling Utility**
```python
from utils import scale_and_save_data

train_scaled, test_scaled, scaler = scale_and_save_data(
    'data/processed/train.csv',
    'data/processed/test.csv',
    'model/scaler.pkl',
    columns_to_scale=['Close', 'Open', 'High', 'Low', 'Volume']
)
```

**Dashboard Screenshot:**  
![Dashboard Screenshot](image/demo_dashboard.png)

---

## Testing

- **Run tests:**
  ```bash
  pytest
  ```
- **Linting:**
  ```bash
  flake8 .
  ```
- **Test Coverage:**
  ```bash
  pytest --cov=.
  ```

---

## Contributing

1. **Fork** this repository
2. **Create a branch** (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add new feature'`)
4. **Push** to your branch (`git push origin feature/your-feature`)
5. **Open a Pull Request**  
   Please follow [Conventional Commits](https://www.conventionalcommits.org/) and PEP8 code style.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Authors & Contact

- **Manh Hoang** ([23092003e](https://github.com/23092003e))  
  📧 hoangvanmanh2309@gmail.com

For questions, suggestions, or issues, please [open an issue](https://github.com/23092003e/Stock-Price-Forecasting-System-Using-Apache-Spark/issues) or contact via email.

---

> **Disclaimer:** This application is for educational purposes only. Stock price predictions are not guaranteed and should not be used as financial advice.

    
