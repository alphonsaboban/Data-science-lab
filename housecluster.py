import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# ============================
# (a) EXPLORATORY DATA ANALYSIS
# ============================
data = pd.read_csv("/content/Melbourne_housing_FULL.csv")
print("First 5 rows:\n", data.head())
print("\nInfo:\n", data.info())
print("\nStatistical Summary:\n", data.describe())
print("\nMissing Values per Column:\n", data.isnull().sum())
# Drop rows with NaN values across the whole dataset to keep alignment
data = data.dropna()
# Extract numeric columns
numeric_data = data.select_dtypes(include=[np.number])
print("\nNumeric Columns Used:\n", numeric_data.columns.tolist())
# ============================
# (b) VISUALIZATION
# ============================
# (i) Scatter Plot (example using available numeric features)
if 'Landsize' in data.columns and 'Price' in data.columns:
plt.figure(figsize=(8,6))
sns.scatterplot(x='Landsize', y='Price', data=data)
plt.title("Scatter Plot: Landsize vs Price")
plt.show()
# (ii) Histogram
numeric_data.hist(figsize=(12,8), bins=20)
plt.suptitle("Histograms of Housing Features")
plt.show()
# (iii) Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(data=numeric_data)
plt.title("Box Plot of Housing Features")
plt.show()In [4]:
# ============================
# (c) K-MEANS CLUSTERING
# ============================
# Standardize numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_data)
# Try K values for Elbow Method
inertia = []
K = range(1, 11)
for k in K:
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)
inertia.append(kmeans.inertia_)
# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
# Choose optimal K (from elbow)
optimal_k = 3
# Apply K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
# Add cluster labels back — now lengths match
data['Cluster'] = clusters
print("\nCluster counts:\n", data['Cluster'].value_counts())
# Visualize clusters (if applicable)
if 'Landsize' in data.columns and 'Price' in data.columns:
plt.figure(figsize=(8,6))
sns.scatterplot(x='Landsize', y='Price', hue='Cluster', data=data, palette=
plt.title(f"K-Means Clustering (k={optimal_k}) - Landsize vs Price")
plt.show()
# Save results
data.to_csv("Housing_Price_Clustered.csv", index=False)
print("\nClustered dataset saved as 'Housing_Price_Clustered.csv'")
