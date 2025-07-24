# 📰 News Article Classification

## 🔍 Overview
The spread of fake news on the internet is a big problem today. This project aims to create a machine-learning model that can tell whether a news article is fake or real. We use different techniques to analyze and understand the text in these articles. Our dataset includes articles that are already labeled as 'fake' or 'real', which helps us train and test our model.

## 📝 Objective

The main goal is to accurately classify news articles as either 'fake' or 'real' based on their content. By leveraging the power of machine learning and text analysis, we aim to create a reliable tool that can help mitigate the spread of misinformation.

## 📄 Dataset

The dataset used in this project is sourced from the [Dataset](https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset). It includes a balanced set of labeled news articles, which consist of the following features:

- **Title**: The headline of the news article.
- **Text**: The full-text content of the news article.
- **Label**: 0 for fake news and 1 for real news.

## 📋 Methodology

### i. Data Preprocessing:

- **Exploratory Data Analysis (EDA)**: Understanding the data distribution, identifying patterns, and visualizing trends.
- **Text Cleaning**: Removing noise, stop words, and irrelevant characters from the text data.
- **Feature Engineering**: Converting text into numerical representations using techniques like TF-IDF, Bag of Words, and word embeddings.

### ii. Model Selection:

- **Algorithm Comparison**: Evaluating the performance of various models such as Logistic Regression, Random Forest, Decision Tree, Gradient Boost, and Support Vector Machines (SVM).
- **Hyperparameter Tuning**: Optimizing model parameters through techniques like Grid Search and Cross-Validation to enhance model performance.

### iii. Model Evaluation:

- **Evaluation Metrics**: Using metrics such as accuracy, precision, recall, F1-score, and ROC AUC to assess model performance.
- **Confusion Matrix**: Analyzing the confusion matrix to understand the model's ability to distinguish between fake and real news.

### iv. Deployment:

- **Prediction Script**: Creating a Python script to predict whether a new article is fake or real.
- **Web Interface (Optional)**: Developing a simple web application using Streamlit for news classification.

## ✅ Requirements

- **Python 3.x**
- **Libraries**: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, Streamlit etc.

## 🌐 The Web Interface
<img width="1431" height="700" alt="Screenshot 2025-07-24 at 19 01 57" src="https://github.com/user-attachments/assets/a748ee46-27b4-44b4-bde6-eee6853c7c51" />

## 🧪 Testing Model
i. **For True News** 
<img width="1379" height="708" alt="Screenshot 2025-07-24 at 19 04 00" src="https://github.com/user-attachments/assets/d987d181-ea2b-47df-a05c-c4f2cf3da9b1" />


ii. **For Fake News**
<img width="1345" height="662" alt="Screenshot 2025-07-24 at 19 16 14" src="https://github.com/user-attachments/assets/32cd997b-db3f-4682-a82d-5bd000d8f309" />



## 📁 Files in the Repository

Certainly! Here’s a simple description of each file and folder in the repository:

### i. README.md

This is the file you’re currently reading. It provides an overview of the project, how to set it up, use it, and other essential information about the NewsArticleClassification project.

### ii. Visualizations Folder

This folder contains various charts, graphs, and plots that help visualize data insights and the performance of our model. You’ll find:
- Histograms and scatter plots showing the distribution of data.
- Word clouds illustrating the most common words in fake and real news articles.
- ROC curves and confusion matrices that display how well our model is performing.

### iii. app.py

This Python script is the main application file for predicting whether news articles are fake or real. It uses the trained machine learning model to classify new articles based on their text content. 

### iv. fake.csv

This file contains the data of fake news articles used for training and testing our model. Each row represents an article labeled as fake, along with its title and content.

### v. true.csv

This file holds the data of real news articles. Similar to `true.csv`, it includes articles that are labeled as real, along with their title and content.

### vi. FakeNewsPredictor.ipynb

This is a Jupyter notebook that includes the complete code for our project. It covers:
- Data preprocessing steps like cleaning and preparing the text.
- Feature extraction methods to convert text into a format suitable for machine learning.
- Training and evaluating different machine learning models to find the best one for detecting fake news.

### vii. manual_testing.csv

This file is used for manually testing our model. It contains a small set of news articles that you can use to see how well the model predicts fake or real news. It’s helpful for checking the model’s performance on new, unseen data.




