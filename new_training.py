import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
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

# Create MultiOutputRegressor for Gradient Boosting model
gb_model = MultiOutputRegressor(GradientBoostingRegressor())
gb_model.fit(X_train, y_train)

# Create MultiOutputRegressor for KNN model
knn_model = MultiOutputRegressor(KNeighborsRegressor())
knn_model.fit(X_train, y_train)

# Make predictions
gb_preds = gb_model.predict(X_test)
knn_preds = knn_model.predict(X_test)

# Compute evaluation metrics for Gradient Boosting model
gb_mse = mean_squared_error(y_test, gb_preds)
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_r2 = r2_score(y_test, gb_preds)
gb_rmse = np.sqrt(gb_mse)

# Compute evaluation metrics for KNN model
knn_mse = mean_squared_error(y_test, knn_preds)
knn_mae = mean_absolute_error(y_test, knn_preds)
knn_r2 = r2_score(y_test, knn_preds)
knn_rmse = np.sqrt(knn_mse)

# Print evaluation metrics
print("Gradient Boosting Evaluation Metrics:")
print("Mean Squared Error (MSE):", gb_mse)
print("Mean Absolute Error (MAE):", gb_mae)
print("R-squared (R2) Score:", gb_r2)
print("Root Mean Squared Error (RMSE):", gb_rmse)

print("\nKNN Evaluation Metrics:")
print("Mean Squared Error (MSE):", knn_mse)
print("Mean Absolute Error (MAE):", knn_mae)
print("R-squared (R2) Score:", knn_r2)
print("Root Mean Squared Error (RMSE):", knn_rmse)

# Visualize predictions
plt.figure(figsize=(12, 6))

# Scatter plot of actual vs. predicted values for latitude
plt.subplot(1, 3, 1)
plt.scatter(y_test["latitude"], gb_preds[:, 0], label="Gradient Boosting", alpha=0.5)
plt.scatter(y_test["latitude"], knn_preds[:, 0], label="KNN", alpha=0.5)
plt.xlabel("Actual Latitude")
plt.ylabel("Predicted Latitude")
plt.title("Latitude Predictions")
plt.legend()

# Scatter plot of actual vs. predicted values for longitude
plt.subplot(1, 3, 2)
plt.scatter(y_test["longitude"], gb_preds[:, 1], label="Gradient Boosting", alpha=0.5)
plt.scatter(y_test["longitude"], knn_preds[:, 1], label="KNN", alpha=0.5)
plt.xlabel("Actual Longitude")
plt.ylabel("Predicted Longitude")
plt.title("Longitude Predictions")
plt.legend()

# Scatter plot of actual vs. predicted values for altitude
plt.subplot(1, 3, 3)
plt.scatter(y_test["altitude"], gb_preds[:, 2], label="Gradient Boosting", alpha=0.5)
plt.scatter(y_test["altitude"], knn_preds[:, 2], label="KNN", alpha=0.5)
plt.xlabel("Actual Altitude")
plt.ylabel("Predicted Altitude")
plt.title("Altitude Predictions")
plt.legend()

plt.tight_layout()
plt.show()

# Visualize performance metrics
plt.figure(figsize=(10, 6))

# Bar plot of MSE for both models
plt.subplot(2, 2, 1)
plt.bar(["Gradient Boosting", "KNN"], [gb_mse, knn_mse])
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Comparison")

# Bar plot of MAE for both models
plt.subplot(2, 2, 2)
plt.bar(["Gradient Boosting", "KNN"], [gb_mae, knn_mae])
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE Comparison")

# Bar plot of R2 score for both models
plt.subplot(2, 2, 3)
plt.bar(["Gradient Boosting", "KNN"], [gb_r2, knn_r2])
plt.ylabel("R-squared (R2) Score")
plt.title("R2 Score Comparison")

# Bar plot of RMSE for both models
plt.subplot(2, 2, 4)
plt.bar(["Gradient Boosting", "KNN"], [gb_rmse, knn_rmse])
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("RMSE Comparison")

plt.tight_layout()
plt.show()

# # Function to get user input
# def get_user_input():
#     acc_x = float(input("Enter acceleration in x-direction: "))
#     acc_y = float(input("Enter acceleration in y-direction: "))
#     acc_z = float(input("Enter acceleration in z-direction: "))
#     gyro_x = float(input("Enter angular velocity in x-direction: "))
#     gyro_y = float(input("Enter angular velocity in y-direction: "))
#     gyro_z = float(input("Enter angular velocity in z-direction: "))
#     mag_x = float(input("Enter magnetometer reading in x-direction: "))
#     mag_y = float(input("Enter magnetometer reading in y-direction: "))
#     mag_z = float(input("Enter magnetometer reading in z-direction: "))
#     w = float(input("Enter quaternion value (w): "))
#     linear_vel_x = float(input("Enter linear velocity in x-direction: "))
#     linear_vel_y = float(input("Enter linear velocity in y-direction: "))
#     angular_vel_z = float(input("Enter angular velocity in z-direction: "))
#     orientation = float(input("Enter orientation: "))
    
#     # Create a DataFrame with user input
#     user_data = pd.DataFrame({
#         "acc_x": [acc_x],
#         "acc_y": [acc_y],
#         "acc_z": [acc_z],
#         "gyro_x": [gyro_x],
#         "gyro_y": [gyro_y],
#         "gyro_z": [gyro_z],
#         "mag_x": [mag_x],
#         "mag_y": [mag_y],
#         "mag_z": [mag_z],
#         "w": [w],
#         "linear_vel_x": [linear_vel_x],
#         "linear_vel_y": [linear_vel_y],
#         "angular_vel_z": [angular_vel_z],
#         "orientation": [orientation]
#     })
    
#     return user_data

# # Get user input
# user_data = get_user_input()

# # Use trained models to make predictions
# gb_prediction = gb_model.predict(user_data)
# knn_prediction = knn_model.predict(user_data)

# # Print predictions
# print("\nPredicted Latitude (Gradient Boosting):", gb_prediction[0][0])
# print("Predicted Longitude (Gradient Boosting):", gb_prediction[0][1])
# print("Predicted Altitude (Gradient Boosting):", gb_prediction[0][2])

# print("\nPredicted Latitude (KNN):", knn_prediction[0][0])
# print("Predicted Longitude (KNN):", knn_prediction[0][1])
# print("Predicted Altitude (KNN):", knn_prediction[0][2])
