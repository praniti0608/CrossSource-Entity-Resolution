# Amazon ML Challenge 2026 — Business Entity Resolution

A machine learning project focused on identifying matching business entities across multiple data sources.

## Overview

Business records often contain inconsistent names, addresses, formatting, and missing values. This project explores how machine learning and text-processing techniques can identify records that refer to the same real-world business.

The pipeline works with three sources of business records and combines preprocessing, candidate generation, feature engineering, and supervised classification.

## Technology Stack

- Python
- Pandas and NumPy
- LightGBM
- Scikit-learn
- Joblib
- Unidecode

## Pipeline Architecture

1. **Data preprocessing**
   - Text normalization
   - Unicode-aware transliteration
   - Business name and address processing

2. **Candidate generation and blocking**
   - Generate potential matching record pairs
   - Reduce the search space before classification

3. **Feature engineering**
   - Extract comparison features
   - Represent similarities between candidate records

4. **Model training**
   - LightGBM-based matching classifier
   - Supervised match prediction

5. **Prediction and evaluation**
   - Generate entity matching predictions
   - Evaluate using ROC-AUC and Average Precision

## Project Structure

```text
amazon-ml-challenge-2026/
├── Code/
│   └── business_entity_resolution/
│       └── src/
│           ├── preprocessing.py
│           ├── blocking.py
│           ├── candidate_generation.py
│           ├── candidate_generation_fast.py
│           ├── features.py
│           ├── feature_generation_sample.py
│           ├── train.py
│           ├── train_matcher_sample.py
│           ├── predict.py
│           ├── evaluation.py
│           ├── integrated_pipeline.py
│           ├── data_loader.py
│           ├── data_audit.py
│           ├── dataset_deep_profile.py
│           ├── candidate_miss_diagnostic.py
│           ├── candidate_filter_diagnostic.py
│           └── test_hit_counter.py
├── requirements.txt
├── .gitignore
└── README.md

```
