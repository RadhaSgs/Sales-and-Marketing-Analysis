
# Real-World Sales and Marketing Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Real_World_Project_Data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Visualize spending score by gender
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='SpendingScore', data=data)
plt.title('Spending Score by Gender')
plt.show()

# Correlation analysis (numeric columns only)
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze marketing campaign response
response_rate = data['CampaignResponse'].mean() * 100
print(f"Campaign Response Rate: {response_rate:.2f}%")

# Campaign ROI calculation
data['ROI'] = (data['Sales'] - data['CampaignCost']) / data['CampaignCost']
print("\nCampaign ROI:")
print(data[['CustomerID', 'ROI']])

# Visualize ROI by region
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='ROI', data=data, ci=None)
plt.title('ROI by Region')
plt.show()
