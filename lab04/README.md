# Lab 4: Titanic Fare Prediction Using Regression

### Author: Eric Meyer
### Date: 4/4/2025

## Overview
This project explores various regression techniques to predict passenger **fare** from the Titanic dataset.  
We use linear, polynomial, and regularized regression models to evaluate how well features like **age**, **passenger class**, and **family size** explain variation in ticket prices.

## Project Components
- Data Cleaning & Feature Engineering  
- Exploratory Data Analysis (EDA)  
- Model Training:
  - Linear Regression  
  - Ridge Regression  
  - ElasticNet  
  - Polynomial Regression (up to degree 8)  
- Model Evaluation & Comparison  
- Visualizations & Interpretation  

## Dataset
We use the Titanic dataset available from the Seaborn library. Key features selected:
- `age`: Passenger age  
- `pclass`: Ticket class (1st, 2nd, 3rd)  
- `sibsp` and `parch`: Used to create `family_size`  
- `fare`: Target variable (continuous)

Additional preprocessing:
- Missing values imputed or removed  
- Created new feature: `family_size = sibsp + parch + 1`

## Results

### Model Comparison

| Model             | R² Score | RMSE   | MAE   |
|------------------|----------|--------|--------|
| Linear            | 0.317    | 31.44  | 20.70  |
| Ridge             | 0.317    | 31.43  | 20.69  |
| ElasticNet        | **0.352**| 30.61  | 19.61  |
| Polynomial (deg 3)| 0.346    | 30.76  | **18.52** |

- **Best Overall:** ElasticNet performed best based on R² and RMSE.
- **Best MAE:** Polynomial (degree 3) captured lower average error, suggesting better performance on smaller fare values.

## Visualizations
- Scatter plots of actual vs predicted fares  
- Polynomial regression curves (degrees 3 through 8)  
- Comparison plots of error metrics across models  

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ericmeyer1/applied-ml-eric.git


