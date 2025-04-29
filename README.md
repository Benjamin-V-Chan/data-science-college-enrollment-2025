# data-science-college-enrollment-2025

# README.md

## Project Overview

This project analyzes detailed Fall term enrollment data from Montgomery College, focusing on student demographics, academic programs, geographic background, and campus attendance patterns. It includes preprocessing, feature engineering, exploratory data analysis, clustering, predictive modeling, and model interpretation. The goal is to uncover hidden patterns in student enrollment and predict primary campus choice using machine learning techniques.

## Folder Structure

```
project-root/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── outputs/
│   ├── eda/
│   ├── clustering/
│   ├── predictive_model/
│   └── model_interpretation/
├── models/
│   └── encoders/
├── requirements.txt
└── README.md
```
## Usage

1. Setup the Project:

Clone the repository.
Ensure you have Python installed.
Install required dependencies using the requirements.txt file.
```bash
pip install -r requirements.txt
```

2. Preprocess the raw data:
```bash
python scripts/01_data_preprocessing.py
```

3. Perform feature engineering:
```bash
python scripts/02_feature_engineering.py
```

4. Conduct exploratory data analysis:
```bash
python scripts/03_exploratory_data_analysis.py
```

5. Perform clustering analysis:
```bash
python scripts/04_clustering.py
```

6. Train predictive models:
```bash
python scripts/05_predictive_modeling.py
```

7. Visualize model interpretations:
```bash
python scripts/06_model_interpretation_visualization.py
```

