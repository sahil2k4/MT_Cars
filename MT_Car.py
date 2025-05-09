import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Load Data
@st.cache
def load_data():
    df = pd.read_csv('../DataSets/MTCARS.csv')
    df = df.rename(columns={'Unnamed: 0': 'Model'})
    return df

df = load_data()

# Data Preprocessing
df['Model'] = df['Model'].astype(str)  # Convert to string for labeling
X = df.drop('Model', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

# Display Key Insights Header
st.title("Key Insights from Clustering of MTCARS Dataset")

# Key Insight on Clustering
st.write("""
### Cluster Insights:
1. **Cluster Separation (Feature Averages)**:
   - Different clusters have distinct average characteristics based on key features like **MPG (fuel efficiency)**, **Weight (WT)**, and **Horsepower (HP)**.
   - This segmentation helps identify car types like **economy cars**, **sports cars**, and **luxury cars**.

2. **Visualizing Cluster Separation**:
   - **PCA Visualization** shows how cars are grouped based on their features.
   - Color-coded clusters represent different car types.

3. **Cluster Features**:
   - Cluster 0: Economy cars (high MPG, low weight)
   - Cluster 1: Luxury cars (high HP, high weight)
   - Cluster 2: Mid-range cars (balanced MPG, moderate weight)
   - Cluster 3: Heavy-duty cars (low MPG, high horsepower)
""")

# Show Cluster Feature Averages
cluster_summary = df.groupby('Cluster').mean()

# Display Cluster Summary as a Table
st.write("### Cluster Feature Averages")
st.dataframe(cluster_summary.round(1))  # Rounded for better readability

# Cluster Averages Histogram
st.write("### Distribution of Cluster Feature Averages")
fig_summary = px.bar(cluster_summary.T, 
                     labels={"value": "Average", "index": "Feature"},
                     title="Cluster Feature Averages")
st.plotly_chart(fig_summary)

# PCA Plot: Scatterplot of Clusters in 2D Space
st.write("### PCA Reduced Cluster Visualization")
fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Model'],
                     title="Cluster Visualization in 2D (PCA-reduced)")
st.plotly_chart(fig_pca)

# Conclusion of Insights
st.write("""
### Purpose of Clustering:
- **Segmentation of Cars**: Helps categorize cars into groups such as **economy**, **luxury**, **sports**, and **heavy-duty**.
- **Decision Making**: Allows users to choose cars based on their needs: fuel efficiency, performance, or load capacity.
- **Simplifies Comparison**: By grouping similar cars, users can easily compare features within the same cluster.
""")
