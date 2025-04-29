import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=['Fall Term'])
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    df = df.replace('Unknown', pd.NA)
    return df

def derive_primary_campus(df):
    campuses = ['Attending Germantown', 'Attending Rockville', 'Attending Takoma Park/SS']
    df['campus_list'] = df[campuses].apply(
        lambda row: [c for c,v in zip(campuses,row) if v=='Yes'], axis=1
    )
    def pick_campus(lst):
        if len(lst)==1: return lst[0]
        if len(lst)>1: return 'Multiple'
        return 'None'
    df['Primary Campus'] = df['campus_list'].apply(pick_campus)
    return df.drop(columns=['campus_list'])

def save_cleaned_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    raw_path   = 'data/raw/Montgomery_College_Enrollment_Data_20250414.csv'
    clean_path = 'data/processed/cleaned_data.csv'
    df = load_data(raw_path)
    df = clean_data(df)
    df = derive_primary_campus(df)
    save_cleaned_data(df, clean_path)

if __name__ == '__main__':
    main()
