import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
walmart = pd.read_csv('C:\\Users\\eldeeb\\OneDrive\\Desktop\\mlflow\\walmart_sales_forecasting.csv')
walmart.drop(columns=['Unnamed: 0', 'total_price'], inplace=True)
walmart['DateTime'] = pd.to_datetime(walmart['DateTime'])
walmart = walmart[walmart['DateTime'].dt.year != 2019]

# Resample weekly profit
weekly_sales = walmart.resample('W-SUN', on='DateTime')['profit'].sum().reset_index()
weekly_sales.rename(columns={'DateTime': 'ds', 'profit': 'y'}, inplace=True)

# Log transform
weekly_sales['y_log'] = np.log1p(weekly_sales['y'])

# Create features
weekly_sales['week'] = weekly_sales['ds'].dt.isocalendar().week
weekly_sales['month'] = weekly_sales['ds'].dt.month
weekly_sales['year'] = weekly_sales['ds'].dt.year

# Train/test split
train = weekly_sales[:-12]
test = weekly_sales[-12:]

X_train = train[['week', 'month', 'year']]
y_train = train['y_log']
X_test = test[['week', 'month', 'year']]
y_test = test['y_log']

# -------------------------------
# 2. Start MLflow experiment
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")  # Or use remote URI
mlflow.set_experiment("Walmart Sales Forecasting")

with mlflow.start_run(run_name="XGBoost Basic Model"):

    # Params
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3

    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_train_pred_log = model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)
    y_train_true = np.expm1(y_train)

    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
    y_test_true = np.expm1(y_test)

    # Metrics
    train_r2 = r2_score(y_train_true, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    test_r2 = r2_score(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

    # Log metrics
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_rmse", test_rmse)

    # Log model
    mlflow.xgboost.log_model(model, artifact_path="xgb_model", registered_model_name="walmart_sales_forecast_model")

    # Save plot
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sales['ds'], weekly_sales['y'], label='Actual')
    plt.plot(test['ds'], y_test_pred, label='Forecast (Test)', color='red')
    plt.title('Weekly Profit Forecast (XGBoost with Log Transform)')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = "forecast_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
