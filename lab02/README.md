### **Machine Learning Project: Titanic Survival Prediction**  
**Author**: Eric Meyer  
**Date**: 03/19/2025  
**Objective**: Build a classification model to predict passenger survival on the Titanic based on key features.

---

#### **Project Overview**  
In this project, we analyze the **Titanic Dataset** to predict passenger survival based on features such as **age, fare, passenger class (pclass), sex, and family size**. We apply **classification models** and compare different train/test splitting techniques.

---

#### **Key Steps**  
1. Data Exploration & Visualization
2. Feature Selection: Age, Fare, Pclass, Sex, Family Size
3. Train/Test Split (Basic vs. Stratified)
4. Model Training: Logistic Regression (baseline model)
5. Model Evaluation: Accuracy, Precision, Recall, F1-score

---

#### **Results**  
##### **Class Distribution**
- **Original Dataset:** (Survived: 38.4%, Not Survived: 61.6%)
- **Basic Train/Test Split:** (Train: 38.9%, Test: 36.3%)
- **Stratified Train/Test Split:** (Train: 38.3%, Test: 38.5%)

##### **Train/Test Splitting Comparison**  
- The **basic train/test split** introduced imbalance, affecting the test set.
- The **stratified split** preserved class distributions, leading to a better-balanced dataset.

---

#### **Learnings**  
- Explored the importance of **feature selection** in classification problems.
- Understood the impact of **stratified sampling** vs. **random train/test split**.
- Learned how to handle **imbalanced datasets** to improve model fairness.
- Gained experience in **classification metrics** beyond accuracy.
- Strengthened data preprocessing and exploratory data analysis (EDA) skills.

---

#### **Tools Used**  
- **Python**: For data processing, model training, and evaluation.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: For numerical operations and calculations.
- **Matplotlib & Seaborn**: Utilized for data visualization.
- **Scikit-learn**: Key library for classification models and performance evaluation.
- **Jupyter Notebook**: Interactive environment for coding, analysis, and visualization.

