# Adult Census Income Prediction

### Project Overview

The objective of this project is to build a **machine learning model** that predicts whether an individual earns **more than $50K per year**, based on demographic and socio-economic characteristics.

This project addresses a **supervised binary classification problem** using the **Adult Census Income dataset**, a widely used dataset in machine learning research and education.

The work is developed as part of the **Machine Learning Zoomcamp (DataTalksClub)** and follows best practices for data analysis, modeling, and reproducibility.

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


