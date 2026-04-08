📌 Fake Job Postings Detection using Machine Learning

 Project Name

Fake Job Postings Detection using NLP and Machine Learning

📖 What This Project Does

This project detects whether a job posting is real or fake using machine learning techniques.

It analyzes job description text and classifies job posts using:

TF-IDF Vectorization
Logistic Regression Model

This helps in identifying fraudulent job advertisements automatically.

⚙️ Steps Used in This Project

Load dataset using Pandas
Data preprocessing (handle missing values)
Combine text columns:
title
description
requirements
Convert text into numerical format using TF-IDF
Split dataset into training and testing sets
Train Logistic Regression model
Evaluate model using:
Accuracy Score
Classification Report
Confusion Matrix

Visualize results using Seaborn heatmap

📊 Dataset Information

Dataset name: fake_job_postings.csv
Type: Job postings dataset
Columns used:
title → Job title
description → Job description
requirements → Job requirements
fraudulent → Target label (0 = real, 1 = fake)

🧰 Libraries Used

pandas → Data loading and manipulation
numpy → Numerical operations
matplotlib → Data visualization
seaborn → Heatmap and plots
scikit-learn → Machine learning (TF-IDF + Logistic Regression)
