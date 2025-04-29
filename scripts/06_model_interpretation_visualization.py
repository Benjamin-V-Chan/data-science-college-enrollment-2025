import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

def load_model(path):
    return joblib.load(path)

def get_feature_names(features_path):
    df = pd.read_csv(features_path)
    return df.drop(columns=[
        'Primary Campus','Primary Campus Enc','Age Group',
        'MC Program Description','MCPS High School','City in MD','ZIP'
    ]).columns.tolist()

def plot_feature_importances(model, names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig = plt.figure()
    plt.barh(range(len(idx)), imp[idx])
    plt.yticks(range(len(idx)), [names[i] for i in idx])
    plt.title('Top 20 Feature Importances')
    fig.savefig(os.path.join(output_dir,'feature_importances.png'))
    plt.close(fig)

def main():
    model_path    = 'models/random_forest_primary_campus.joblib'
    features_path = 'data/processed/features.csv'
    out_dir       = 'outputs/model_interpretation'
    m = load_model(model_path)
    names = get_feature_names(features_path)
    plot_feature_importances(m, names, out_dir)

if __name__ == '__main__':
    main()
