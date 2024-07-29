import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = './customer_churn_dataset-testing-master.csv'
churn_data = pd.read_csv(file_path)

# List of columns part of the dataset
expected_columns = [
    'CustomerID', 'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
    'Last Interaction', 'Churn'
]

# checking for missing columns 
missing_columns = set(expected_columns) - set(churn_data.columns)
if missing_columns:
    print(f"Warning: The following columns are missing: {missing_columns}")

# Checking for missing values in the dataset
missing_values = churn_data.isnull().sum()
print("\nMissing values in each column:\n", missing_values[missing_values > 0])

# Handle missing values, for simplicity, we're just going to drop them
churn_data = churn_data.dropna()

# Perform necessary data transformations
le = LabelEncoder()#convert categorical labels to numerica form
scaler = StandardScaler()#removing mean and scaling to unit variance

for column in churn_data.columns:
    if column == 'CustomerID':
        continue  # Skip CustomerID
    elif churn_data[column].dtype == 'object':
        # Encode categorical variables
        churn_data[column] = le.fit_transform(churn_data[column])
    elif column in ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']:
        # Scale numerical variables
        churn_data[column] = scaler.fit_transform(churn_data[column].values.reshape(-1, 1))

# Save processed information to new file
output_file_path = './processed_customer_churn_data.csv'
churn_data.to_csv(output_file_path, index=False)
print(f"\nProcessed data saved to: {output_file_path}")

# Display the first few rows of the processed data
print("\nFirst few rows of processed data:")
print(churn_data.head())

# Provide summary statistics of the dataset
print("\nSummary statistics:\n", churn_data.describe())

# Creating visualizations
plt.figure(figsize=(12, 6))

# Distribution of Churn
plt.subplot(121)
sns.countplot(x='Churn', data=churn_data)
plt.title('Distribution of Churn')

# Correlation between Age and Churn
plt.subplot(122)
sns.boxplot(x='Churn', y='Age', data=churn_data)
plt.title('Age vs Churn')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(churn_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
