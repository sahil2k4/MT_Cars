import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Set up Streamlit page configuration
st.set_page_config(page_title="MTCARS Clustering App", layout="wide")
st.title("🚗 MTCARS Clustering App (KMeans + PCA)")

# File uploader
uploaded_file = st.sidebar.file_uploader("📤 Upload your MTCARS CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.rename(columns={"Unnamed: 0": "Model"})
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("🔍 Raw Dataset")
    st.dataframe(df)

    # Encode model name
    label = LabelEncoder()
    df['Model'] = label.fit_transform(df['Model'])

    # Scale the data
    X_scaled = StandardScaler().fit_transform(df.drop('Model', axis=1))

    # Elbow Method to determine optimal k
    inertia = []
    k_range = range(1, 15)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    st.subheader("📈 Elbow Curve")
    fig_elbow = px.line(x=list(k_range), y=inertia, markers=True, labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
    fig_elbow.update_layout(title="Elbow Method for Choosing k")
    st.plotly_chart(fig_elbow)

    # Hardcoding k=4 instead of user input (you can choose any k value)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    score = silhouette_score(X_scaled, df['Cluster'])
    st.sidebar.markdown(f"**Silhouette Score:** {score:.2f}")

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]

    st.subheader("🌀 PCA Cluster Visualization")
    fig_pca = px.scatter(
        df, x='PCA1', y='PCA2', color=df['Cluster'].astype(str), hover_data=['Model'],
        title="Clusters visualized with PCA", color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_pca)

    # Cluster feature averages with rounding to 1 decimal place
    st.subheader("📊 Cluster Feature Averages")
    cluster_summary = df.groupby('Cluster').mean().reset_index()

    # Round the values to 1 decimal place
    cluster_summary_rounded = cluster_summary.drop('Cluster', axis=1).round(1)

    # Show annotated heatmap
    fig_summary = ff.create_annotated_heatmap(
        z=cluster_summary_rounded.values,
        x=list(cluster_summary_rounded.columns),
        y=[f"Cluster {i}" for i in cluster_summary['Cluster']],
        colorscale='Viridis',
        showscale=True
    )
    st.plotly_chart(fig_summary)

    # Display Short Key Insights for Cluster Feature Averages
    st.subheader("🔑 Key Insights from Cluster Feature Averages")

    # Example of insights based on the cluster averages
    insights = """
    - **Cluster 0** tends to have **lower mpg** (miles per gallon) and **higher weight** (wt), indicating it's likely a cluster of **heavier cars**.
    - **Cluster 1** shows **higher horsepower** (hp) and **larger displacement** (disp), possibly **sports cars** or **high-performance vehicles**.
    - **Cluster 2** displays **high mpg** and **low weight**, suggesting it's a cluster of **economy cars** or **fuel-efficient vehicles**.
    - **Cluster 3** is characterized by **medium weight and mpg**, possibly indicating **mid-range cars**.
    - Features like **gear count** (gear) and **carburetors** (carb) seem to define step-like groupings, which could indicate **categorical divisions** based on car type (e.g., 4-cylinder vs 8-cylinder).
    """

    st.markdown(insights)

else:
    st.info("👈 Please upload the `MTCARS.csv` file to begin.")
