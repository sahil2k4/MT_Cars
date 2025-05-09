import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

st.set_page_config(page_title="MTCARS Clustering", layout="wide")
st.title("🚗 MTCARS Clustering App (KMeans + PCA)")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your MTCARS CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.rename(columns={"Unnamed: 0": "Model"})  # Rename index column
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.subheader("🔍 Raw Dataset")
    st.dataframe(df)

    # Encode model
    label = LabelEncoder()
    df['Model'] = label.fit_transform(df['Model'])

    # Scaling
    X_scaled = StandardScaler().fit_transform(df.drop('Model', axis=1))

    # KMeans Elbow Method
    inertia = []
    k_range = range(1, 15)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    st.subheader("📈 Elbow Curve")
    fig_elbow = px.line(x=list(k_range), y=inertia, markers=True, labels={'x': 'K', 'y': 'Inertia'})
    fig_elbow.update_layout(title="Elbow Method to Choose Optimal K")
    st.plotly_chart(fig_elbow)

    # Select K value
    k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df['Cluster'])
    st.sidebar.markdown(f"**Silhouette Score:** {score:.2f}")

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = pca_comp[:, 0], pca_comp[:, 1]

    st.subheader("🌀 Cluster Visualization (PCA)")
    fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Model'], title="KMeans Clusters (PCA Reduced)")
    st.plotly_chart(fig_pca)

    # Cluster Summary
    st.subheader("📊 Cluster Feature Averages")
    cluster_summary = df.groupby('Cluster').mean().reset_index()
    fig_summary = ff.create_annotated_heatmap(
        z=cluster_summary.drop('Cluster', axis=1).values,
        x=cluster_summary.columns[1:],
        y=[f"Cluster {i}" for i in cluster_summary['Cluster']],
        colorscale='RdBu',
        showscale=True
    )
    st.plotly_chart(fig_summary)

    st.subheader("📁 Clustered Data Sample")
    st.dataframe(df.head())

else:
    st.info("📤 Please upload the `MTCARS.csv` file to get started.")
