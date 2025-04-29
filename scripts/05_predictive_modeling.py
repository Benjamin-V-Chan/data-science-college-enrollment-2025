import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_features(path):
    return pd.read_csv(path)

def prepare_data(df):
    y = df['Primary Campus Enc']
    X = df.drop(columns=[
        'Primary Campus','Primary Campus Enc','Age Group',
        'MC Program Description','MCPS High School','City in MD','ZIP'
    ])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X_train, y_train)
    return m

def evaluate_model(m, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = m.predict(X_test)
    cr = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(cr).transpose().to_csv(os.path.join(output_dir,'classification_report.csv'))
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=m.classes_, columns=m.classes_).to_csv(os.path.join(output_dir,'confusion_matrix.csv'))

def save_model(m, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(m, path)

def main():
    feat_path   = 'data/processed/features.csv'
    model_path  = 'models/random_forest_primary_campus.joblib'
    output_dir  = 'outputs/predictive_model'
    df = load_features(feat_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    m = train_model(X_train, y_train)
    save_model(m, model_path)
    evaluate_model(m, X_test, y_test, output_dir)

if __name__ == '__main__':
    main()
