# Predictive Maintenance using Machine Learning

An end-to-end machine-learning project using the **AI4I 2020 Predictive Maintenance Dataset** to explore machine condition data, predict machine failures, investigate failure modes, compare anomaly-detection methods, and explain model predictions.

> **Dataset note:** AI4I 2020 is a synthetic dataset. Results in this repository are experimental results on this dataset and should not be interpreted as validated performance on real industrial equipment.

## Project objectives

- Explore machine sensor measurements and failure patterns.
- Prepare features for machine-failure classification.
- Compare supervised classification models.
- Investigate class imbalance and decision thresholds.
- Predict individual failure modes as a multi-label task.
- Compare unsupervised anomaly-detection methods.
- Explain model behaviour using feature importance and SHAP.
- Demonstrate a Streamlit prediction interface.

## Dataset

The AI4I 2020 dataset contains 10,000 observations and 14 original columns:

`UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

The five failure-mode labels are:
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

The binary classification target is `Machine failure`.

### Leakage prevention

The five failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were excluded from the input features when predicting `Machine failure`. They were retained for the separate multi-label failure-mode experiment.

## Workflow

1. Dataset loading and understanding
2. Exploratory Data Analysis
3. Data cleaning and preprocessing
4. Feature engineering
5. Binary machine-failure classification
6. Imbalanced-data experiments
7. Model evaluation and threshold experiments
8. Multi-label failure-mode prediction
9. Anomaly detection
10. Explainable AI
11. Streamlit demonstration

## Feature engineering

Two derived features were used:

- **Temperature Difference** = Process temperature − Air temperature
- **Mechanical Power** = Torque × angular speed, where angular speed is converted from rpm to radians per second

Mechanical Power is derived from existing measurements and is not an independent sensor reading.

## Binary classification results

The reported held-out test split contained 2,000 observations, including 68 machine failures and 1,932 non-failures.

### Reported model comparison

| Model / configuration | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9675 | 0.5714 | 0.1765 | 0.2697 |
| Decision Tree | 0.9820 | 0.7424 | 0.7206 | 0.7313 |
| Random Forest | 0.9905 | 0.9455 | 0.7647 | 0.8455 |
| XGBoost | 0.9890 | 0.9423 | 0.7206 | 0.8167 |
| Random Forest + SMOTE | 0.9790 | 0.6585 | 0.7941 | 0.7200 |
| Class-weighted Random Forest | 0.9890 | 0.8966 | 0.7647 | 0.8254 |

The baseline Random Forest had the strongest F1 score among the listed model configurations at the default 0.50 threshold.

### Exploratory threshold experiment

For the original Random Forest, a threshold of 0.30 produced:

- Precision: 0.8769
- Recall: 0.8382
- F1: 0.8571
- Confusion matrix: `[[1924, 8], [11, 57]]`

The threshold was explored on the test set during development. For a rigorous final evaluation, threshold selection should be performed on a validation set and the test set should remain untouched until final evaluation. Therefore, the threshold-0.30 result is exploratory and should not be presented as an unbiased final estimate.

## Multi-label failure-mode prediction

A separate multi-output Random Forest experiment predicted `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.

Reported baseline per-label F1 scores:
- TWF: 0.00 (10 positive test cases)
- HDF: 0.91 (29 positive test cases)
- PWF: 1.00 (13 positive test cases)
- OSF: 0.87 (16 positive test cases)
- RNF: 0.00 (4 positive test cases)

Rare failure labels had very few positive examples. The results for TWF and RNF are weak and highly uncertain; perfect scores on a small support should not be assumed to generalize.

## Anomaly detection results

Anomaly detectors were trained using normal training observations and evaluated against the labelled test set. An anomaly is not necessarily the same thing as a confirmed machine failure.

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.9415 | 0.2118 | 0.2647 | 0.2353 |
| Local Outlier Factor (LOF) | 0.9460 | 0.3148 | 0.5000 | 0.3864 |
| One-Class SVM | 0.9415 | 0.2525 | 0.3676 | 0.2994 |

LOF was the strongest of these three anomaly-detection methods by the reported F1 score, but its performance remained below the supervised Random Forest classifier.

## Explainability

The project used:
- Random Forest feature importance
- SHAP TreeExplainer
- SHAP summary plots
- SHAP bar plots
- Individual prediction explanations

Feature importance describes model reliance, not causation. SHAP values explain model outputs, not physical causal mechanisms.

## Streamlit application

The Streamlit interface accepts:
- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

It derives the engineered features, applies the saved encoder and scaler, and displays the predicted class and estimated failure probability.

### Run locally

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the locally generated model artifacts beside `app.py`:
   - `predictive_maintenance_model.pkl`
   - `scaler.pkl`
   - `encoder.pkl`

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

**Important:** The model artifacts are not included in this repository bundle. Generate them from your own notebook using the same training feature order and preprocessing steps. Do not commit model files unless you have checked their provenance, licensing, and repository-size implications.

## Repository structure

```text
Predictive-Maintenance-ML/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── README.md
├── reports/
│   └── figures/
│       └── README.md
└── src/
    └── README.md
```

The submitted notebook is included at `notebooks/Smart Industrial Predictive Maintenance System.ipynb`. Twenty-six PNG figures were extracted from its saved cell outputs into `reports/figures/`.

## Limitations

- AI4I 2020 is synthetic; performance may not transfer to real production machinery.
- The dataset has substantial class imbalance.
- Some failure modes have very few positive examples, making per-label metrics unstable.
- Anomaly detection identifies unusual observations, not necessarily confirmed failures.
- The exploratory 0.30 decision-threshold result was selected using the test set and requires re-evaluation with validation-based threshold selection.
- The current Streamlit app is a demonstration and is not a safety-critical monitoring system.
- No real-time sensor integration, alerting, maintenance scheduling, or industrial validation is included.
- Remaining useful life (RUL) is not predicted because this dataset does not provide suitable true time-to-failure labels.

## Future improvements

- Rebuild preprocessing and the estimator as one scikit-learn `Pipeline` / `ColumnTransformer`.
- Use a train/validation/test strategy for model and threshold selection.
- Evaluate precision-recall curves, calibration, and cost-sensitive operating points.
- Collect more positive examples for rare failure modes.
- Validate on real, time-ordered machine data from an industrial setting.
- Add model monitoring, drift detection, logging, and alert handling.
- Add automated tests and input validation to the Streamlit application.
- Consider a separate, properly labelled dataset for remaining useful life prediction.
