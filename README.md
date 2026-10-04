# ⚡ Electricity Theft Detection

A Machine Learning project that analyzes electricity consumption patterns and identifies consumers whose usage appears potentially suspicious.

> **Note:** This project is for educational and analytical purposes. A model prediction does not prove actual electricity theft and should be verified using appropriate utility and field-level information.

## 📌 Project Overview

Electricity theft can lead to financial losses for electricity distribution companies.

This project uses historical electricity consumption data and Machine Learning classification algorithms to identify potentially suspicious consumption patterns.

The project includes:

- Data cleaning and preprocessing
- Missing-value handling
- Feature and target separation
- Train-test splitting
- Multiple classification models
- Model evaluation
- Random Forest model selection
- Streamlit web application
- Prediction probability display

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Git & GitHub

## 📊 Dataset

The dataset contains electricity consumption information for thousands of consumers across multiple dates.

### Dataset Statistics

- Total consumers: **42,372**
- Original features: **1,036 columns**
- Features used for ML: **1,032**
- Target variable: **FLAG**
- `FLAG = 0` → Normal
- `FLAG = 1` → Potentially suspicious

The dataset contains a large number of daily electricity consumption features.

The raw and cleaned datasets are not uploaded to GitHub because of their large file size.

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Exploration
     ↓
Missing Value Analysis
     ↓
Remove Highly Missing Columns
     ↓
Fill Missing Values
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Random Forest Model
     ↓
Streamlit Application
