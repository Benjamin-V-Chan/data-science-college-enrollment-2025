# - import pandas, os
# - define load_data(path): read raw CSV into DataFrame
# - define clean_data(df): 
#     • drop constant or irrelevant columns
#     • strip whitespace from object columns
#     • replace 'Unknown' with NaN
# - define derive_primary_campus(df):
#     • for each row, collect campuses where value=='Yes'
#     • if exactly one campus → that campus; if multiple → 'Multiple'; else 'None'
#     • drop intermediate list column
# - define save_cleaned_data(df, path): make dirs, write CSV
# - in main(): call above in sequence, using data/raw/… and data/processed/… paths