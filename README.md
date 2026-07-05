# Nairobi Air Quality ARMA Forecast 

A custom-built environmental time-series forecasting application designed to process multi-station sensor feeds and predict urban air quality and meteorological shifts across Nairobi, Kenya. The core engine is built from scratch using raw matrix algebra to model complex temporal relationships.

---

## Stack, Tools & Libraries Used

This project is built using a minimal, highly optimized scientific Python stack to maintain direct control over memory layout and matrix calculations:

* **Python 3.12+**: Core development environment.
* **NumPy**: The absolute backbone of the project. Used for handling raw multi-dimensional matrix layouts, execution of the matrix dot products ($\mathbf{X} \cdot \mathbf{W} + \mathbf{b}$), vector slicing, manual gradient tracking, and element-wise matrix math.
* **Pandas**: Powering the data engineering pipeline. Used for importing raw sensor streams, one-hot encoding geographical features, sorting chronologically, and running isolated grouping mechanics.
* **Streamlit**: Direct interactive frontend deployment wrapper. Used to translate the model's raw output vectors into a production-ready dashboard web application.
* **Plotly**: Interactive visualization library integrated into the dashboard to plot historical atmospheric profiles, error residuals, and forecast traces.

---

## What the Engine Does & Core Features

The software acts as a specialized data pipeline and a predictive matrix engine that handles data from ingestion to production visualization: