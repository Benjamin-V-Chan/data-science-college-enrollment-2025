# - import pandas, os, joblib, sklearn.preprocessing
# - load cleaned data from data/processed/cleaned_data.csv
# - select categorical columns to one-hot encode (e.g. Gender, Race, etc.)
# - fit OneHotEncoder(handle_unknown='ignore') → array → DataFrame → concat
# - drop original categorical columns
# - label-encode 'Age Group' and 'Primary Campus'
# - save features to data/processed/features.csv
# - dump encoders to models/encoders/