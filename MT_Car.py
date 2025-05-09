import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚗 MTCARS Clustering with KMeans + Plotly + Streamlit")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../DataSets/MTCARS.csv')
    df = df.rename(columns={'Unnamed: 0': 'Model'})
    return df

df = load_data()

# Encode model names
label = LabelEncoder()
df['Model_Label'] = label.fit_transform(df['Model'])

# Sidebar settings
st.sidebar.header("🔧 KMeans Settings")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

# Drop label and scale
features = df.drop(['Model', 'Model_Label'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

# Elbow curve
inertia = []
k_range = range(1, 15)
for i in k_range:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

elbow_fig = go.Figure()
elbow_fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
elbow_fig.update_layout(title="Elbow Method: Optimal k", xaxis_title="Number of Clusters", yaxis_title="Inertia")

# Silhouette Score
sil_score = silhouette_score(X_scaled, df['Cluster'])

# PCA scatter plot
pca_fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Model'], 
                     title="Cluster Visualization (PCA Reduced)", color_continuous_scale='Set2')

# Cluster Summary
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)

# Heatmap
st.subheader("📊 Elbow Method & Silhouette Score")
st.plotly_chart(elbow_fig, use_container_width=True)
st.write(f"🧮 Silhouette Score for k={k}: **{sil_score:.2f}**")

st.subheader("🔍 PCA Cluster Visualization")
st.plotly_chart(pca_fig, use_container_width=True)

st.subheader("📌 Cluster Summary (Average Values)")
st.dataframe(cluster_summary.style.highlight_max(axis=0), height=400)

st.subheader("🔥 Heatmap of Cluster Averages")
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_summary.T, annot=True, cmap='coolwarm')
st.pyplot(plt.gcf())

# Optional: Show raw data
with st.expander("🔎 Show Raw Data"):
    st.dataframe(df)
