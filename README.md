🚢 Sonar-Based Rock vs Mine Detection (Machine Learning)

This project implements a Machine Learning–based sonar detection system to classify underwater objects as Rock or Mine using Logistic Regression. The model is trained on real-world sonar signal data and evaluated using accuracy metrics.

🔍 Project Overview

Binary classification using Logistic Regression

Dataset consists of sonar signal readings

Proper train-test split with stratification applied

Performance evaluated on both training and testing data

Supports real-time prediction through user-provided sonar input

📊 Model Performance

Training Accuracy: 84.40%

Testing Accuracy: 76.19%

🛠 Tech Stack

Python

NumPy

Pandas

Scikit-Learn

📁 Dataset

The dataset contains multiple numerical features representing sonar signal strengths, with labels indicating:

R → Rock

M → Mine

🚀 Future Improvements

Integrate real sonar sensors with Arduino to build a real-time detection system

Replace manual input with live sensor data streaming

Experiment with advanced models (SVM, Random Forest, Neural Networks)

Deploy the model as a web or embedded system application
