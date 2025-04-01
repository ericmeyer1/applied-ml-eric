# Midterm: **Banknote Authentication Using Machine Learning**

### Author: Eric Meyer
### Date: 3/26/2025
### Project Notebook: [ML Midterm Folder](https://github.com/ericmeyer1/applied-ml-eric/tree/main/midterm)

## **Overview**
This project applies machine learning techniques to the **Banknote Authentication Dataset** in order to classify banknotes as either authentic or fraudulent. The dataset consists of various features extracted from banknotes, and the goal is to predict whether a given banknote is genuine (class 0) or fake (class 1). In this project, we explored the **Decision Tree** and **Random Forest** classification models to achieve the best possible performance in predicting banknote authenticity.

---

## **Project Files**
- 📄 **[Jupyter Notebook](https://github.com/ericmeyer1/applied-ml-eric/blob/main/midterm/ml_midterm_eric.ipynb)** – Full implementation with code and explanations.  
- 📄 **[Peer Review](https://github.com/SchroderJ-pixel/ml-04-mushroom-classification/blob/main/classifiation_schroder.ipynb)** – Review feedback and reflections. 

1. Clarity & Organization

Strengths: The notebook is well-structured with clear sections, including data exploration, feature engineering, and model evaluation. Markdown cells are used effectively to explain each step.

Areas for Improvement: Some sections could benefit from more detailed explanations. For instance, the transition between data preprocessing and feature selection could be smoother with an introductory summary.

2. Feature Selection & Justification

Strengths: The chosen features align well with the project objectives. There is a logical progression from raw data to engineered features.

Areas for Improvement: While the features make sense, a brief justification for why certain features were included or excluded would improve transparency. Consider adding a correlation heatmap or feature importance plot to support these choices.

3. Model Performance & Comparisons

Strengths: The notebook presents multiple models and provides performance metrics for each, allowing for a meaningful comparison.

Areas for Improvement: The performance results are clear, but additional visualizations (e.g., ROC curves or residual plots) could further illustrate the differences between models. Additionally, a discussion on why one model outperforms another would enhance the analysis.

4. Reflection Quality

Strengths: The conclusions summarize key insights effectively, highlighting what worked well and potential limitations.

Areas for Improvement: A section discussing potential future improvements (e.g., additional features, hyperparameter tuning, or alternative modeling approaches) would be beneficial.

Overall Feedback

The notebook is well-structured and demonstrates a solid approach to data analysis and modeling. Enhancing feature justification, adding visual comparisons for model performance, and expanding the reflection section would further improve the overall quality. Great work, Justin!

---

## **Setup Instructions**
To run this project locally, follow these steps:

### **1. Clone the Repository**
git clone https://github.com/ericmeyer1/applied-ml-eric.git
cd applied-ml-eric

---

## **Data Description**

The **Banknote Authentication Dataset** consists of 1372 instances and 5 columns:
- **variance**: The variance of wavelet transformed image.
- **skewness**: The skewness of wavelet transformed image.
- **curtosis**: The curtosis of wavelet transformed image.
- **entropy**: The entropy of wavelet transformed image.
- **class**: The target variable, where `0` represents a genuine banknote and `1` represents a fake banknote.

---

## **Libraries Used**

- `pandas`: For data manipulation and analysis.
- `matplotlib`: For visualizations (e.g., histograms, boxplots).
- `seaborn`: For enhanced visualizations and plotting.
- `scikit-learn`: For model training and evaluation (e.g., decision tree, random forest, metrics).
- `ucimlrepo`: For fetching the dataset from the UCI repository.

---

## **Steps Followed**

### **1. Data Loading and Exploration**
The dataset was fetched from the UCI Machine Learning Repository and loaded into a pandas DataFrame. We explored the dataset to identify any missing values and performed initial summary statistics.

### **2. Data Preprocessing**
We checked for any missing data, but there were no missing values in the dataset. To improve model performance, we applied **data standardization** using `StandardScaler`, which scaled the features to have zero mean and unit variance.

### **3. Model Training**
Two classification models were applied:
1. **Decision Tree Classifier**: A basic yet effective model that splits data based on feature values.
2. **Random Forest Classifier**: An ensemble method that builds multiple decision trees to improve accuracy and reduce overfitting.

### **4. Model Evaluation**
We evaluated both models using the following metrics:
- **Accuracy**: Percentage of correctly predicted instances.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Confusion Matrix**: Visual representation of true positives, true negatives, false positives, and false negatives.

### **5. Results**
- The **Decision Tree model** achieved an accuracy of **99.2%**.
- The **Random Forest model** outperformed the Decision Tree with an accuracy of **99.64%**.

---

## **Evaluation Metrics**

### **Decision Tree Results**
- **Accuracy**: 99.27%  
- **Classification Report**:  
  - **Precision**: 1.00 (class 0), 0.98 (class 1)  
  - **Recall**: 0.99 (class 0), 1.00 (class 1)  
  - **F1-Score**: 0.99 (class 0), 0.99 (class 1)  
- **Confusion Matrix**: Few misclassifications, but high accuracy in distinguishing between genuine and fake banknotes.

### **Random Forest Results**
- **Accuracy**: 99.64%  
- **Classification Report**:  
  - **Precision**: 1.00 (class 0), 0.99 (class 1)  
  - **Recall**: 0.99 (class 0), 1.00 (class 1)  
  - **F1-Score**: 1.00 (class 0), 1.00 (class 1)  
- **Confusion Matrix**: Nearly perfect classification with almost no misclassifications.

---

## **Reflection**
This project reinforced my understanding of classification models, particularly the strengths of **Decision Trees** for this type of problem. It also highlighted the importance of data preprocessing, model selection, and the use of ensemble methods to improve performance. Through this project, I gained valuable experience in evaluating models using metrics like accuracy, precision, recall, and F1-score.