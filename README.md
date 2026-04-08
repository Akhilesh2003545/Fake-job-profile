Fake Job Postings Detection using Machine Learning
Project Name

Fake Job Postings Detection using NLP and Machine Learning

📖 What This Project Does

This project detects whether a job posting is real or fake using machine learning techniques.
It analyzes job description text and classifies postings using TF-IDF vectorization and Logistic Regression.

⚙️ Steps Used in This Project
Load dataset using Pandas
Data preprocessing (handling missing values)
Combine text columns (title + description + requirements)
Convert text into numerical features using TF-IDF
Split data into training and testing sets
Train Logistic Regression model
Evaluate model using:
Accuracy Score
Classification Report
Confusion Matrix
Visualize results using Seaborn heatmap
📊 Dataset Information
Dataset: fake_job_postings.csv
Columns used:
title
description
requirements
fraudulent (target label)
🧰 Libraries Used
pandas → Data handling
numpy → Numerical operations
matplotlib → Visualization
seaborn → Data visualization
scikit-learn → Machine learning (TF-IDF + Logistic Regression)
