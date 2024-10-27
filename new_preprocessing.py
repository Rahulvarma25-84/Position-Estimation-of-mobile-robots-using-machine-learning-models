import pandas as pd

# Load data from files
imu_data = pd.read_csv("imu_log.txt", header=None, names=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z", "w", "timestamp"])
odom_data = pd.read_csv("odom_log.txt", header=None, names=["linear_vel_x", "linear_vel_y", "linear_vel_z", "angular_vel_x", "angular_vel_y", "angular_vel_z", "orientation", "timestamp"])
gps_data = pd.read_csv("RTK_log.txt", header=None, names=["latitude", "longitude", "altitude", "timestamp"])

# Drop timestamp column from each dataset
imu_data.drop(columns=["timestamp"], inplace=True)
odom_data.drop(columns=["timestamp"], inplace=True)
gps_data.drop(columns=["timestamp"], inplace=True)

# Drop the columns with all zeros
odom_data.drop(columns=["linear_vel_z", "angular_vel_x", "angular_vel_y"], inplace=True)

# Merge datasets
merged_data = pd.concat([imu_data, odom_data, gps_data], axis=1)

# Drop rows with missing values
merged_data.dropna(inplace=True)

# Save the merged dataset to a new file
merged_data.to_csv("new_merged_dataset.csv", index=False)
