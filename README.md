# fairness-analysis
## What Influences the Perception of Fairness in Urban and Rural China? An Analysis Using Machine Learning

This repository contains a Python script for training a Gradient Boosting Regressor model and analyzing feature importance using SHAP (SHapley Additive exPlanations). The script follows a structured pipeline including data preprocessing, model training, and visualization of feature importance.

## Dataset Information

This code is part of the research for the paper “What Influences the Perception of Fairness in Urban and Rural China? An Analysis Using Machine Learning.” The dataset used in this study is the Chinese Social Survey (CSS), a large-scale longitudinal sampling survey initiated by the Institute of Sociology at the Chinese Academy of Social Sciences in 2005. The purpose of the CSS is to collect data on labor and employment, family and social life, social attitudes, and other aspects of the Chinese public, providing a comprehensive resource for studying social changes in transitional China. The dataset serves as a valuable foundation for social science research and government decision-making.

For more details, visit the official CSS website: http://css.cssn.cn/css_sy/.

## Features
- Loads and preprocesses a dataset from a CSV file.
- Handles missing values and infinite values.
- Splits the dataset into training and testing sets.
- Trains a Gradient Boosting Regressor model.
- Computes SHAP values for feature importance analysis.
- Visualizes feature importance using bar charts and summary plots.

## Requirements
To run the script, ensure you have the following dependencies installed:

```bash
pip install shap pandas numpy matplotlib scikit-learn
```

## Usage

1. Place the dataset (`data.csv`) in the appropriate directory and update the file path in the script if necessary.
2. Run the script using Python:

```bash
python script.py
```

3. The script will output:
   - A bar chart displaying the top 20 most important features.
   - A SHAP summary plot visualizing feature impact.

## Data Preprocessing
- Fills missing numeric values with the column mean.
- Fills missing categorical values with the mode.
- Replaces infinite values with NaN and removes missing values.

## Model Training
- Uses `GradientBoostingRegressor` from `sklearn.ensemble`.
- Splits data into training (80%) and testing (20%) subsets.
- Fits the model to the training data.

## Feature Importance Analysis
- Utilizes SHAP's `TreeExplainer` to calculate SHAP values.
- Extracts and visualizes the top 20 features based on mean absolute SHAP values.
- Generates two plots: a horizontal bar chart and a SHAP summary plot.

## Visualization
- The bar chart uses a color map to indicate feature importance ranking.
- The SHAP summary plot provides an in-depth view of how features influence predictions.


## Contact
This project is maintained by Yating Ding. If you have any questions regarding the code or need further details, please contact via email: vicky-dyt@whu.edu.cn.

For any contributions, please open an issue or submit a pull request.

## Environment
The code is designed to run in a Python 3.8 environment.


