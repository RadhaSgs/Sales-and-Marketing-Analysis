
# Real-World Sales and Marketing Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load the dataset
data = pd.read_csv('Enhanced_Real_World_Data.csv')

# Preview the dataset
print("Dataset Preview:")
print(data.head())

# -------------------------
# STEP 1: Customer Segmentation (K-Means Clustering)
# -------------------------
print("\nStep 1: Performing Customer Segmentation using K-Means Clustering...")

# Prepare data for clustering
cluster_data = data[["AnnualIncome", "SpendingScore"]]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(cluster_data)

# Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data['AnnualIncome'], y=data['SpendingScore'], hue=data['Cluster'],
    palette='viridis', s=100
)
plt.title("Customer Segmentation (Clusters)")
plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.savefig("Customer_Segmentation_Clusters.png")
plt.show()

# -------------------------
# STEP 2: Predictive Modeling (Linear Regression)
# -------------------------
print("\nStep 2: Predicting Sales using Linear Regression...")

# Define features (X) and target (y)
X = data[["AnnualIncome", "SpendingScore", "CampaignCost"]]
y = data["Sales"]

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Make Predictions
data['PredictedSales'] = model.predict(X)

# Display Model Coefficients
print("Model Coefficients:")
print(f"AnnualIncome: {model.coef_[0]}")
print(f"SpendingScore: {model.coef_[1]}")
print(f"CampaignCost: {model.coef_[2]}")

# -------------------------
# STEP 3: ROI Analysis
# -------------------------
print("\nStep 3: Calculating and Visualizing ROI by Region...")

# Calculate ROI
data['ROI'] = (data['Sales'] - data['CampaignCost']) / data['CampaignCost']

# Visualize ROI by Region
plt.figure(figsize=(8, 6))
sns.barplot(x=data['Region'], y=data['ROI'], ci=None, palette="coolwarm")
plt.title("ROI by Region")
plt.xlabel("Region")
plt.ylabel("Return on Investment (ROI)")
plt.savefig("ROI_by_Region.png")
plt.show()

# -------------------------
# Save Enhanced Dataset
# -------------------------
data.to_csv("Enhanced_Real_World_Data_With_Clusters.csv", index=False)
print("\nEnhanced dataset saved as 'Enhanced_Real_World_Data_With_Clusters.csv'.")
