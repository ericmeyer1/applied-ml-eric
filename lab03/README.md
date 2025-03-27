# Lab 3: Titanic - Predicting Survival

### Author: Eric Meyer
### Date: 3/26/2025

## Introduction

In this project, we predict survival on the Titanic using various machine learning models. The dataset includes various features such as passenger age, sex, class, and more. We will train and evaluate three common classification models: Decision Tree, Support Vector Machine (SVM), and Neural Network (NN). The goal is to assess the performance of each model and determine which factors most influence the prediction of survival.

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
