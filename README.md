# Credit Card Fraud Detection

A Streamlit demo app for identifying potentially fraudulent credit card transactions using an XGBoost model.

## Features

- Select a fraud, legitimate, or random transaction.
- View transaction details, fraud probability, and classification results.
- Review feature contributions and session detection history.
- View model evaluation metrics in the sidebar.

## Notebooks

- **Exploratory Data Analysis** — explore the dataset and identify key patterns.
- **Data Preprocessing** — prepare and process data for modeling.
- **Model Building** — develop and train the XGBoost model.
- **Testing and Evaluation** — test the model and evaluate its performance.

## Dataset

Dataset: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).

The app uses `data/samples.csv` as its source of demo transactions. The file is sampled from the validation and test sets so demo records are not taken from the training data. Since features `V1`–`V28` are anonymized PCA components, this app is an inference demo, not a form for scoring new transactions.

## Model

The model uses the **XGBoost** algorithm to classify credit card transactions.

Optimal Threshold: **0.8328**

Model performance metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.9310 |
| Recall | 0.7606 |
| F1-Score | 0.8372 |
| ROC-AUC | 0.9715 |
| PR-AUC | 0.7878 |
| False Alarm Rate | 0.0001 |

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python tests/run_all.py
```

Tests cover classification logic and feature engineering, model and artifact integration, and the Streamlit UI.