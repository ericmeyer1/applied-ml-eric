# Lab 5: Wine Quality Prediction Using Ensemble Models

### Author: Eric Meyer
### Date: 4/9/2025

## Overview
This project uses ensemble machine learning techniques to predict the **quality of red wine** based on physicochemical test results.  
By comparing models like **Random Forest** and **Gradient Boosting**, we explore how tree-based methods handle a multiclass classification problem with nonlinear patterns and complex feature interactions.

## Project Components
- Data Preprocessing & Feature Standardization  
- Exploratory Data Analysis (EDA)  
- Model Training & Tuning:
  - Random Forest Classifier  
  - Gradient Boosting Classifier  
- Model Evaluation:
  - Accuracy & F1 Score Comparison  
  - Overfitting Detection via Train/Test Gaps  
- Conclusions & Insights  

## Dataset
The dataset comes from the UCI Machine Learning Repository and contains **red wine quality** ratings (scores from 3–8) and the following features:

- `fixed acidity`, `volatile acidity`, `citric acid`  
- `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`  
- `density`, `pH`, `sulphates`, `alcohol`  
- `quality`: Target variable (ordinal multiclass)

Preprocessing steps:
- Removed outliers for improved model stability  
- Scaled features using `StandardScaler`  
- Stratified split into training and test sets  

## Results

### Model Comparison

| Model                   | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Accuracy Gap | F1 Score Gap |
|------------------------|----------------|---------------|----------|---------|---------------|--------------|
| Random Forest (100)    | 1.0000         | 0.8875        | 1.0000   | 0.8661  | 0.1125        | 0.1339       |
| Gradient Boosting (100)| 0.9601         | 0.8563        | 0.9584   | 0.8411  | 0.1039        | 0.1173       |

- **Best Accuracy & F1:** Random Forest achieved the highest performance but showed stronger signs of overfitting.  
- **More Stable:** Gradient Boosting had slightly lower metrics but better generalization potential.

## Visualizations
- Confusion matrices for both models  
- Feature importance plots (Random Forest & Gradient Boosting)  
- Bar charts comparing accuracy and F1 scores across models  
- Train vs Test performance gap visuals

## Conclusions & Insights
- Ensemble models significantly outperform simpler classifiers on this dataset due to their ability to model **nonlinear relationships** and **interactions**.
- **Random Forest** performed best in raw metrics but was highly overfit to the training set.
- **Gradient Boosting** is more tunable and showed greater promise for scalability and generalization with hyperparameter tuning.

## Future Improvements
If this were a competition or real deployment scenario, I would:

- Perform hyperparameter tuning (e.g., `max_depth`, `learning_rate`)  
- Try advanced boosters like **XGBoost** or **LightGBM**  
- Use **SHAP values** to improve model explainability  
- Incorporate **class balancing techniques** (SMOTE or weighting)  
- Use **cross-validation** to reduce performance variance  
- Engineer domain-driven features (e.g., `alcohol/sulphates` ratio)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ericmeyer1/applied-ml-eric.git
