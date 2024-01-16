# Cardiovascular disease prediction using classification algorithms

## Technologies Used

**Language:** Python.

**Libraries:** numpy, pandas, matplotlib, seaborn, pickle, collections, sklearn, xgboost

------------

## 1. Introduction

 [In this work](https://github.com/UrkoRegueiro/Machine_Learning_Projects/blob/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/cardiovascular_disease_prediction-GIT.ipynb) we analyze and explore a dataset on cardiovascular disease. The purpose of this project is to predict whether or not a pacient has cardiovascular disease.

Data set link: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## 2. Exploratory Data Analysis

All of the dataset values were collected at the moment of medical examination.

Data description:

There are 3 types of input features:

- 1- Objective Features (factual information):
	- age: Age of the pacient(days) | int
	- height: Height of the pacient(cm) | int
	- weight: Weight of the pacient(kg) | float
	- gender: Gender of the pacient | boolean

- 2- Examination Feature(results of medical examination):
	- ap_hi: Systolic blood pressure(mm Hg) | int
	- ap_lo: Diastolic blood pressure(mm Hg) | int
	- cholesterol: Cholesterol | categorical | 1: normal, 2: above normal, 3: well above normal
	- gluc: Glucose | categorical | 1: normal, 2: above normal, 3: well above normal

- 3- Subjective Feature(information given by the patient):
	- smoke: Smoking pacient | boolean
	- alco: Alcohol intake pacient | boolean
	- active: Physical activity | boolean

- Target variable:
	- cardio: Presence or absence of cardiovascular disease | boolean

### 2.1. Checking for duplicates

We didn't find duplicates in the dataset.

### 2.2. Outliers
We observed potential outliers in the maximum values of some variables:

#### 2.2.1. "ap_hi" Column
![](https://raw.githubusercontent.com/UrkoRegueiro/Machine_Learning_Projects/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/ap_hi.png)
#### 2.2.2. "ap_lo" Column
![](https://raw.githubusercontent.com/UrkoRegueiro/Machine_Learning_Projects/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/ap_lo.png)
#### 2.2.3. "height" Column
![](https://raw.githubusercontent.com/UrkoRegueiro/Machine_Learning_Projects/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/height.png)
#### 2.2.4. "weight" Column
![](https://raw.githubusercontent.com/UrkoRegueiro/Machine_Learning_Projects/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/weight.png)

------------


Let's first address the outliers in the 'ap_hi' and 'ap_lo' columns, as they are the most significant. To do this, we will consider the following:

The European Society of Cardiology divides blood pressure levels into three categories:
- Optimal: Systolic pressure less than 120 mmHg and diastolic pressure less than 80 mmHg.
- Normal: Systolic pressure between 120-129 mmHg and/or diastolic pressure between 80-84 mmHg.
- High-normal: Systolic pressure between 130/85 mmHg and/or diastolic pressure between 139/89 mmHg.

Based on these values, three grades of hypertension are defined:
- Grade 1 Hypertension: Systolic pressure 140-159 mmHg and/or diastolic pressure 90-99 mmHg.
- Grade 2 Hypertension: Systolic pressure 160-179 mmHg and/or diastolic pressure 100-109 mmHg.
- Grade 3 Hypertension: Systolic pressure greater than or equal to 180 mmHg and/or diastolic pressure greater than or equal to 110 mmHg.

In this work we'll define outliers as NaN's, so after cleaning the dataframe, we can impute them using KNNImputer.

### 2.3. Checking for class imbalance

There is no class imbalance in this dataset.

### 2.4. Let's analyze the subjective columns

Let's see if there are differences between people with healthy habits and those who have some type of unhealthy habit in relation to cardiovascular problems. If noticeable differences exist, we will keep these columns for further analysis.

- We define a healthy person as someone who does not smoke (0), does not drink (0), and is active (1).
- We define an unhealthy person as someone who has any unhealthy habit.

------------


We observed that people with healthy habits have a lower percentage of cardiovascular problems, although the difference is not significant compared to those with some type of unhealthy habit.
As there was a small difference between habits, we considered the subjective columns for this analysis, as they may contribute predictive information.

## 3. Data Preprocessing

In this section we processed the data to be suitable for our models.

### 3.1. OneHot encoding
### 3.2. Label encoding
### 3.3. Converting the values to the International System of Units
### 3.4. Creation of new variables
### 3.5. Data type transformation
### 3.6. Correlation analysis

We discovered that height DOES NOT seemed to show a correlation with cardiovascular problems, so we removed it.

### 3.7. Column deletion
### 3.8. Imputing values

In this section we converted the values of systolic and diastolic blood pressure to NaN and then imputed them using the KNNImputer.

- For systolic blood pressure ('ap_hi'), values outside the range of 90 to 210 mmHg were converted to NaN.
- For diastolic blood pressure ('ap_lo'), values outside the range of 50 to 140 mmHg were converted to NaN.

### 3.9. Feature Selection
## 4. Training models

In this section we selected the models shown below and we chose the best ones:
- Logistic Regression
- Gaussian NB
- KNeighbors Classifier
- Nearest Centroid
- Random Forest Classifier
- SVC
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGB Classifier
- Hist Gradient Boosting Classifier

### 4.1. Hyperparametrization
#### 4.1.1. Gradient Boosting Classifier
#### 4.1.2. Random Forest Classifier
#### 4.1.3. Hist Gradient Boosting Classifier
#### 4.1.4. XGB Classifier
#### 4.1.5. Suport Vector Machine
## 5. Final results
The performance of models in Section 4 was:

| Model                          | Accuracy | Precision | Recall   |
|--------------------------------|----------|-----------|----------|
| GradientBoostingClassifier     | 0.746144 | 0.766361  | 0.707723 |
| RandomForestClassifier         | 0.719963 | 0.725346  | 0.707436 |
| HistGradientBoostingClassifier | 0.743275 | 0.759222  | 0.712030 |
| XGBClassifier                  | 0.742988 | 0.764008  | 0.702699 |
| SVC                            | 0.742630 | 0.773124  | 0.686334 |

### 5.1. Gradient Boosting Classifier - Best performance

![](https://raw.githubusercontent.com/UrkoRegueiro/Machine_Learning_Projects/main/Supervised_Learning_Projects/3-%20Cardiovascular_Disease_Prediction/cm.png)

<h5 align="center">

| Model                      | Accuracy | Precision | Recall |
|----------------------------|----------|-----------|--------|
| GradientBoostingClassifier | 0.7460   | 0.7662    | 0.7077 |
</h5>

## 6. Conclusion


After experimenting with hyperparameter tuning across various models, we achieved an accuracy of 74.6% with the GradientBoostingClassifier. Considering that the baseline model yields an accuracy of 49.9%, our model has demonstrated a significant improvement in predicting cardiovascular problems. However, further enhancements would be needed to achieve a higher accuracy or, at the very least, a higher recall at the expense of precision.
