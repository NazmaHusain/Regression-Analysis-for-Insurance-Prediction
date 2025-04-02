# Medical Insurance Cost Prediction using Machine Learning

## Overview
This project utilizes machine learning techniques to analyze and predict medical insurance costs based on various factors such as age, BMI, smoking status, number of children and region. The dataset undergoes preprocessing, visualization, and model training to build the predictive model.

## Dataset
The dataset used in this project (`Medical_insurance.csv`) contains the following features:
- `age`: Age of the individual
- `sex`: Gender (converted to numerical values)
- `bmi`: Body Mass Index
- `children`: Number of children
- `smoker`: Smoking status (converted to numerical values)
- `region`: Region of residence (encoded numerically)
- `charges`: Medical insurance cost (target variable)

## Data Preprocessing
- Checked for missing values.
- Converted categorical data (`sex`, `smoker`, `region`) into numerical values.
- Generated visualizations to understand feature relationships.
- Computed correlation heatmap to identify redundant features.

## Exploratory Data Analysis (EDA)
- Line plots to examine relationships between `age`, `bmi`, and `charges`.
- Bar plots for categorical features (`sex`, `region`, `smoker`, `children`) to analyze their effect on charges.
- Heatmap to explore correlations among numerical features.

## Machine Learning Models
Two regression models were trained to predict medical insurance costs:
1. **RandomForestRegressor**
2. **XGBRegressor** (Selected as the best model based on performance)

### Model Training & Evaluation
- Split the dataset into training (80%) and testing (20%) sets.
- Applied **RandomForestRegressor** and **XGBRegressor** models.
- Evaluated models using **R² score**.
- **Cross-validation (10-fold)** was used to select the best model.
- **XGBRegressor** was identified as the best-performing model.

## Model Deployment
- The final trained model (`XGBRegressor`) was saved using **Joblib**.
- The model can be loaded and used for new predictions.

## Usage
To use the trained model for prediction:
```python
import joblib
loaded_model = joblib.load("final_model_XGB.joblib")
new_data = [[23, 1, 32.007, 10, 1, 3]]  # Example input
predicted_charges = loaded_model.predict(new_data)
print(predicted_charges)
```

## Insights
- Medical insurance costs **increase with age and BMI**.
- **Smokers have significantly higher charges** than non-smokers.
- **XGBRegressor outperforms RandomForestRegressor** in prediction accuracy.

## Requirements
- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- joblib

## Conclusion
This project successfully applies machine learning to predict medical insurance costs. The use of feature engineering, exploratory data analysis, and model evaluation techniques ensures accurate predictions and valuable insights into healthcare costs.

