# - import pandas, os, numpy, sklearn (StandardScaler, KMeans), PCA, matplotlib.pyplot
# - load features.csv
# - select numeric columns (drop Primary Campus Enc)
# - scale features, fit KMeans(n_clusters=5), get labels
# - append labels to DataFrame → save to outputs/clustering/clustered_data.csv
# - PCA to 2D, scatter plot colored by cluster → save to outputs/clustering/