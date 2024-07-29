import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = './customer_churn_dataset-testing-master.csv'
churn_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset
print("Initial data:\n", churn_data.head())

# Checking for missing values in the dataset
missing_values = churn_data.isnull().sum()
print("\nMissing values in each column:\n", missing_values[missing_values > 0])

# Handle missing values, for simplicity, we're just going to drop them
churn_data = churn_data.dropna()

# Perform necessary data transformations
# Encoding categorical variables
le = LabelEncoder()
for column in churn_data.columns:
    if churn_data[column].dtype == 'object':
        churn_data[column] = le.fit_transform(churn_data[column])

# Scaling numerical variables
scaler = StandardScaler()
for column in churn_data.columns:
    if churn_data[column].dtype == 'int64' or churn_data[column].dtype == 'float64':
        churn_data[column] = scaler.fit_transform(churn_data[column].values.reshape(-1, 1))

# Exploratory Data Analysis (EDA)
# Provide summary statistics of the dataset
print("\nSummary statistics:\n", churn_data.describe())

# Create visualizations to understand the distribution of features and the target variable
# For simplicity, let's visualize the distribution of the target variable
sns.countplot(x='Churn', data=churn_data)  # Replace 'Churn' with your target variable
plt.show()
