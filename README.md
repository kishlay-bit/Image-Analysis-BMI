# BMI Prediction from Visual Features

A machine learning project focused on predicting Body Mass Index (BMI) using image-based visual features along with tabular biometric information.

## Project Overview

This project explores the relationship between human visual appearance and BMI using classical machine learning techniques. The pipeline combines image feature extraction with metadata such as weight, height, and gender to build predictive regression models.

The project includes:

* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Classical machine learning model training
* Model evaluation and comparison
* Visualization of prediction performance

---

## Dataset

Dataset Used:
Visual BMI Dataset
https://www.kaggle.com/datasets/abidur14004/visual-bmi

The dataset contains:

* Body and facial images
* BMI values
* Weight
* Height
* Gender labels

The dataset is not included in this repository due to size limitations.

---

## Features Used

### Image Features

* HOG (Histogram of Oriented Gradients)
* Color statistics
* Grayscale intensity statistics

### Tabular Features

* Weight (lb)
* Height (in)
* Gender

---

## Models Implemented

The following regression models were trained and evaluated:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. AdaBoost Regressor
6. XGBoost Regressor

---

## Evaluation Metrics

Models were compared using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score
* MAPE (Mean Absolute Percentage Error)

---

## Project Structure

```bash
bmi_prediction/
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_predict_bmi.ipynb
│
├── results/
│   ├── eda_overview.png
│   ├── metric_comparison.png
│   ├── pred_vs_actual.png
│   ├── residual_plots.png
│   └── model_comparison.csv
│
├── datasets/
├── saved_models/
└── README.md
```

---

## Key Workflow

1. Load and clean dataset
2. Generate image-based features
3. Merge image and tabular features
4. Perform person-aware train/test split
5. Train regression models
6. Compare model performance
7. Save best-performing model

---

## Technologies Used

* Python
* OpenCV
* Scikit-learn
* XGBoost
* NumPy
* Pandas
* Matplotlib
* Scikit-image

---

## Results

The project generated:

* Model comparison tables
* Prediction vs actual plots
* Residual analysis graphs
* Feature-based BMI prediction models

Results and visualizations are available in the `results/` directory.

---

## Future Improvements

* Deep learning based CNN models
* Transfer learning using ResNet/EfficientNet
* Hyperparameter optimization
* Real-time BMI prediction interface
* Deployment using Flask/Streamlit

---

## Author

Kishlay
Computer Science and Engineering
BIT Mesra
