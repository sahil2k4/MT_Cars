import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Loaded Successfully!")
        return df
    else:
        st.warning("Please upload the MTCARS dataset!")
        return None

# Streamlit Layout
st.title('MT Cars Clustering and PCA Analysis')

# File Upload
uploaded_file = st.file_uploader("Upload MTCARS dataset", type="csv")

# Load the data
df = load_data(uploaded_file)

if df is not None:
    # Data Preprocessing
    df = df.rename(columns={'Unnamed: 0': 'Model'})
    label = LabelEncoder()
    df['Model'] = label.fit_transform(df['Model'])

    # Standardizing the data (excluding 'Model' column)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('Model', axis=1))

    # KMeans Clustering
    k_range = range(1, 15)
    inertia = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Elbow Plot for Optimal k
    st.subheader("Elbow Method for Optimal Clusters")
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
    fig_elbow.update_layout(title="Elbow Method For Optimal k", xaxis_title="Number of clusters", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow)

    # Select the number of clusters (k)
    k = st.slider('Select number of clusters', min_value=1, max_value=10, value=4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    score = silhouette_score(X_scaled, df['Cluster'])
    st.write(f"Silhouette Score: {score:.2f}")

    # PCA for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    # Scatter Plot for PCA with Cluster
    st.subheader("PCA Visualization of Clusters")
    fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Model'],
                         title="PCA Reduced Clusters")
    st.plotly_chart(fig_pca)

    # Cluster Feature Averages
    cluster_summary = df.groupby('Cluster').mean()

    st.subheader("Cluster Feature Averages")
    # Convert to 1 decimal place for better readability
    cluster_summary = cluster_summary.round(1)
    st.dataframe(cluster_summary)

    # Heatmap of Feature Averages
    st.subheader("Cluster Feature Average Heatmap")
    fig_summary = ff.create_annotated_heatmap(z=cluster_summary.values,
                                              x=cluster_summary.columns,
                                              y=cluster_summary.index.astype(str).tolist(),
                                              colorscale='YlGnBu',
                                              showscale=True)
    st.plotly_chart(fig_summary)

    # Short Key Insights
    st.subheader("Key Insights from Clustering")
    st.markdown("""
    - **Cluster Separation**: Clusters are separated based on key features such as `mpg` (Miles per Gallon) and `wt` (Weight), showing clear patterns. For example, high `mpg` and low `wt` cars are likely to be economy cars.
    - **Power and Engine Features**: Variables like `hp` (Horsepower) and `disp` (Displacement) play a crucial role in distinguishing clusters. Higher horsepower and displacement are typical of sports or performance cars.
    - **Categorical Features**: Features such as `cyl` (cylinders), `gear` (gears), and `carb` (carburetors) show step-like distributions that influence cluster formation.
    - **Minimal Separation**: Features like `drat` (rear axle ratio) or `vs` (engine shape) show minimal separation, and these may not contribute much to clustering.

    **Usefulness of Clustering**:
    - This clustering method helps categorize cars into distinct groups based on their attributes, assisting manufacturers or customers in identifying patterns of car types.
    - The clusters may align with different categories like economy, sports, or luxury cars.
    """)

# End of Streamlit App
