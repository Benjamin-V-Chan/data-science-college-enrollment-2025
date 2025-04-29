import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_cleaned_data(path):
    return pd.read_csv(path)

def plot_distribution(df, col, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    counts = df[col].value_counts(dropna=False)
    fig = plt.figure()
    counts.plot(kind='bar', title=col)
    fig.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
    plt.close(fig)

def plot_campus_correlation(df, output_dir):
    campuses = ['Attending Germantown','Attending Rockville','Attending Takoma Park/SS']
    mat = df[campuses].applymap(lambda x:1 if x=='Yes' else 0).corr()
    fig = plt.figure()
    plt.imshow(mat, aspect='auto')
    plt.colorbar()
    plt.xticks(range(3), campuses, rotation=45)
    plt.yticks(range(3), campuses)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{mat.iloc[i,j]:.2f}", ha='center', va='center')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir,'campus_correlation.png'))
    plt.close(fig)

def main():
    clean_path = 'data/processed/cleaned_data.csv'
    eda_dir    = 'outputs/eda'
    df = load_cleaned_data(clean_path)
    for col in ['Gender','Ethnicity','Race','Age Group','Primary Campus','Attend Day or Evening']:
        plot_distribution(df, col, eda_dir)
    plot_campus_correlation(df, eda_dir)

if __name__ == '__main__':
    main()
