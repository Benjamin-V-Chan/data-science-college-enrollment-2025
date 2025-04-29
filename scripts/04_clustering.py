import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_features(path):
    return pd.read_csv(path)

def select_features(df):
    return df.select_dtypes(include=[np.number]).drop(columns=['Primary Campus Enc'])

def run_kmeans(X, n_clusters=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Xs)
    return labels, kmeans, scaler, Xs

def save_clusters(df, labels, path):
    df['Cluster'] = labels
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def plot_clusters(Xs, labels, output_dir):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    fig = plt.figure()
    plt.scatter(coords[:,0], coords[:,1], c=labels, s=10)
    plt.title('KMeans Clusters (PCA)')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir,'clusters_pca.png'))
    plt.close(fig)

def main():
    feat_path       = 'data/processed/features.csv'
    cluster_out_csv = 'outputs/clustering/clustered_data.csv'
    cluster_out_dir = 'outputs/clustering'
    df = load_features(feat_path)
    X = select_features(df)
    labels, _, _, Xs = run_kmeans(X)
    save_clusters(df, labels, cluster_out_csv)
    plot_clusters(Xs, labels, cluster_out_dir)

if __name__ == '__main__':
    main()
