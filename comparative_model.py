import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load merged dataset
merged_data = pd.read_csv("new_merged_dataset.csv")

# Split data into features (X) and target (y)
X = merged_data.drop(columns=["latitude", "longitude", "altitude"])
y = merged_data[["latitude", "longitude", "altitude"]]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(),
}

# Create MultiOutputRegressor for each model
multi_output_models = {name: MultiOutputRegressor(model) for name, model in models.items()}

# Train models and make predictions
predictions = {}
for name, model in multi_output_models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

# Compute evaluation metrics for each model
metrics = {}
for name, preds in predictions.items():
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mse)
    metrics[name] = {"MSE": mse, "MAE": mae, "R2": r2, "RMSE": rmse}

# Print evaluation metrics
for name, metric in metrics.items():
    print(f"{name} Evaluation Metrics:")
    for metric_name, value in metric.items():
        print(f"{metric_name}: {value}")
    print("\n")

# Visualize predictions
plt.figure(figsize=(18, 12))

# Scatter plot of actual vs. predicted values for latitude
plt.subplot(3, 1, 1)
for name, preds in predictions.items():
    plt.scatter(y_test["latitude"], preds[:, 0], label=name, alpha=0.5)
plt.xlabel("Actual Latitude")
plt.ylabel("Predicted Latitude")
plt.title("Latitude Predictions")
plt.legend()

# Scatter plot of actual vs. predicted values for longitude
plt.subplot(3, 1, 2)
for name, preds in predictions.items():
    plt.scatter(y_test["longitude"], preds[:, 1], label=name, alpha=0.5)
plt.xlabel("Actual Longitude")
plt.ylabel("Predicted Longitude")
plt.title("Longitude Predictions")
plt.legend()

# Scatter plot of actual vs. predicted values for altitude
plt.subplot(3, 1, 3)
for name, preds in predictions.items():
    plt.scatter(y_test["altitude"], preds[:, 2], label=name, alpha=0.5)
plt.xlabel("Actual Altitude")
plt.ylabel("Predicted Altitude")
plt.title("Altitude Predictions")
plt.legend()

plt.tight_layout()
plt.show()

# Visualize performance metrics
plt.figure(figsize=(12, 12))

# Bar plot of MSE for all models
plt.subplot(2, 2, 1)
plt.bar(metrics.keys(), [metric["MSE"] for metric in metrics.values()])
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Comparison")

# Bar plot of MAE for all models
plt.subplot(2, 2, 2)
plt.bar(metrics.keys(), [metric["MAE"] for metric in metrics.values()])
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE Comparison")

# Bar plot of R2 score for all models
plt.subplot(2, 2, 3)
plt.bar(metrics.keys(), [metric["R2"] for metric in metrics.values()])
plt.ylabel("R-squared (R2) Score")
plt.title("R2 Score Comparison")

# Bar plot of RMSE for all models
plt.subplot(2, 2, 4)
plt.bar(metrics.keys(), [metric["RMSE"] for metric in metrics.values()])
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("RMSE Comparison")

plt.tight_layout()
plt.show()
