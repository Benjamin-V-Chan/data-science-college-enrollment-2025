import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def load_cleaned_data(path):
    return pd.read_csv(path)

def encode_features(df):
    ohe_cols = [
        'Student Type','Student Status','Gender','Ethnicity','Race',
        'Attend Day or Evening','HS Category','State','County in MD'
    ]
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    arr = ohe.fit_transform(df[ohe_cols])
    ohe_df = pd.DataFrame(arr, columns=ohe.get_feature_names(ohe_cols))
    df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1).drop(columns=ohe_cols)
    le_age = LabelEncoder()
    df['Age Group Enc'] = le_age.fit_transform(df['Age Group'].fillna('Unknown'))
    le_pc  = LabelEncoder()
    df['Primary Campus Enc'] = le_pc.fit_transform(df['Primary Campus'].fillna('None'))
    return df, ohe, le_age, le_pc

def save_features(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_encoders(encoders, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(dir_path, f'{name}.joblib'))

def main():
    clean_path    = 'data/processed/cleaned_data.csv'
    features_path = 'data/processed/features.csv'
    df = load_cleaned_data(clean_path)
    df_feat, ohe, le_age, le_pc = encode_features(df)
    save_features(df_feat, features_path)
    save_encoders({
        'onehot': ohe,
        'le_age': le_age,
        'le_primary_campus': le_pc
    }, 'models/encoders')

if __name__ == '__main__':
    main()
