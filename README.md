# Adult Census Income Prediction

## Problem Description

Income estimation is a common challenge in socio-economic analysis, especially when direct income information is incomplete, unavailable, or expensive to collect.

In this project, the problem is formulated as a **binary classification task**: predicting whether an individual earns **more than $50,000 per year**, based on demographic and employment-related attributes obtained from census data. The target variable `income` represents this threshold and is commonly used in income inequality studies.

The model relies on features such as age, education level, occupation, marital status, working hours, and capital gains/losses. These variables are typically available in large-scale surveys, making the approach applicable in real-world analytical settings.

### Intended Use

A model like this can be used for:
- exploratory socio-economic research
- income distribution and inequality analysis
- identifying key factors associated with higher income levels
- educational purposes to demonstrate an end-to-end machine learning workflow

The objective of this project is not to produce an exact income prediction, but to demonstrate how machine learning techniques can be applied to structured tabular data, from data preparation and modeling to evaluation and deployment.

---

## Dataset

- **Source:** UCI Machine Learning Repository / Kaggle  
- **Dataset name:** Adult Census Income  
- **Number of records:** ~48,000  
- **Target variable:** `income`
  - `<=50K`
  - `>50K`

### Main features:
- Age  
- Education  
- Occupation  
- Marital status  
- Hours per week  
- Gender  
- Native country  
- Capital gain / loss  

Dataset link:  
https://www.kaggle.com/datasets/uciml/adult-census-income

---

## Problem Statement

Given a set of demographic and employment-related attributes, the goal is to **predict whether a person earns more than $50K per year**.

This type of problem is relevant for:
- socio-economic analysis  
- income inequality studies  
- understanding key drivers of income  

---

## Machine Learning Approach

- **Problem type:** Binary Classification  
- **Models explored:**
  - Logistic Regression (baseline)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (optional)
- **Evaluation metrics:**
  - Accuracy
  - Precision
  - Recall
  - ROC AUC

Special focus is placed on:
- categorical feature encoding  
- handling missing values  
- model interpretability  

---


