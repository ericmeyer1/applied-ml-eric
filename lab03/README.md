# Lab 3: Titanic - Predicting Survival

### Author: Eric Meyer
### Date: 3/26/2025

## Introduction

In this project, we predict survival on the Titanic using various machine learning models. The dataset includes various features such as passenger age, sex, class, and more. We will train and evaluate three common classification models: Decision Tree, Support Vector Machine (SVM), and Neural Network (NN). The goal is to assess the performance of each model and determine which factors most influence the prediction of survival.

## Learnings and Results

### 1. **Model Performance Comparison**
In this project, we used three different machine learning models to predict survival on the Titanic dataset: **Decision Tree**, **Support Vector Machine (SVC)**, and **Neural Network (MLP)**. Each model was evaluated on the same dataset, and their performance was compared based on key metrics, including accuracy, precision, recall, and F1-score.

- **Decision Tree:** 
  - Strengths: Easy to interpret and fast to train.
  - Weaknesses: Tended to overfit the data, especially on the training set.
  - Results: The decision tree showed good accuracy but struggled to generalize well on the test data.

- **Support Vector Machine (SVC):**
  - Strengths: Effective at finding the best boundary (hyperplane) for separating classes, especially with complex data.
  - Weaknesses: Computationally expensive and may not perform well on large datasets without careful tuning.
  - Results: The SVC model provided solid performance and worked well with the dataset, especially when using the default RBF kernel.

- **Neural Network (MLP):**
  - Strengths: Capable of handling non-linear relationships and complex patterns in data.
  - Weaknesses: Requires more data and tuning to avoid overfitting.
  - Results: The neural network performed better than the decision tree in terms of test set accuracy but was computationally more expensive to train.

### 2. **Feature Selection and Impact**
Several features were tested as input to the models:
- **Alone:** This feature was a binary indicator of whether a passenger was alone or with family. It showed some correlation with survival, but its impact was limited compared to other features.
- **Age:** Age was a significant feature, as it showed a relationship with survival rates, with younger passengers being more likely to survive.
- **Family Size:** The number of family members aboard also had a notable effect on survival. Larger families appeared to have lower survival rates, possibly due to limited space in lifeboats.

### 3. **Model Performance Insights**
- The **Decision Tree** model performed well on training data but overfitted, indicating that it was too complex for the data without proper pruning.
- The **SVC** model showed the most consistent performance, especially when using the RBF kernel, which can handle non-linear separations effectively.
- The **Neural Network** model outperformed others in accuracy but required careful tuning and was more computationally intensive.

### 4. **Challenges Faced**
- **Overfitting:** The decision tree model was prone to overfitting, as expected, due to its high flexibility. This can be mitigated by tuning hyperparameters such as tree depth.
- **Data Imputation:** Missing values, especially in age and embark_town, needed to be imputed. This might have affected model performance slightly, as imputation methods can introduce bias.
- **Model Tuning:** Each model required careful tuning to achieve the best results. For instance, the SVC model needed kernel adjustments to perform well, and the neural network required proper architecture and solver selection to avoid overfitting.

### 5. **Next Steps**
- **Hyperparameter Tuning:** Further hyperparameter tuning could improve model performance, especially for the decision tree and SVC models.
- **Feature Engineering:** Additional features, such as BMI or socioeconomic class, could be explored to improve predictions.
- **Model Ensembling:** Combining the strengths of different models through ensembling methods like bagging or boosting might provide better performance than using a single model.

## Libraries Used

The following Python libraries are used in this project:

- `seaborn` - For loading and visualizing the Titanic dataset.
- `pandas` - For data manipulation and cleaning.
- `matplotlib` - For plotting and visualizations.
- `sklearn` - For implementing classification models and evaluation metrics.

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns





