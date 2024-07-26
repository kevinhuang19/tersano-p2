import pandas as pd

# Load the dataset
file_path = './customer_churn_dataset-testing-master.csv'  # Update with your file path
churn_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial data:\n", churn_data.head())

# Check for missing values
missing_values = churn_data.isnull().sum()
print("\nMissing values in each column:\n", missing_values[missing_values > 0])
