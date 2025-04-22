# ✈️ Travel Route Planner - Flight Delay Prediction & Route Optimization with Machine Learning

This project presents a machine learning-based flight route optimization system designed to minimize flight delays, travel time, and environmental impact. By leveraging multiple machine learning models and real flight data, the system predicts potential delays and suggests optimized flight paths between airports.

# 📌 Project Overview

Goal: Predict flight delays and suggest the most efficient flight route using historical data, distance, weather conditions, and other features.
Techniques Used: Supervised Machine Learning (classification & regression), Graph Theory (for shortest path analysis), Feature Engineering.
Models Implemented:
Linear Regression
Logistic Regression
Decision Tree Classifier
Naïve Bayes
Support Vector Machine (SVM)

# 🧠 Core Features

Predicts whether a flight will be delayed or not using different ML algorithms.
Classifies the type of delay (e.g., weather-related, carrier-related, etc.).
Constructs a weighted graph of airports and computes the shortest and least-delayed route.
Evaluates models using metrics such as Accuracy, F1-Score, Mean Squared Error (MSE), and Cross-Validation Score.

# 📊 Dataset

Source: flights.csv
Features:
AvgTicketPrice
Origin and Destination airports
OriginWeather and DestWeather
DistanceKilometers, FlightTimeHour, FlightDelayMin
dayOfWeek, hour_of_day, Cancelled, and others

# 🧪 Model Performance Summary

![eval](https://github.com/user-attachments/assets/febe92e0-1cb1-47fc-92a9-6f8445b83fe4)

✅ Best overall model for delay prediction: Linear Regression
✅ Best classifier for delay type: Decision Tree

# 🗺️ Route Optimization

Airports are represented as graph nodes
Flights are modeled as edges weighted by delay likelihood + distance
Dijkstra’s algorithm is used to determine the most efficient route

# 📈 Evaluation Metrics

Accuracy (Train, Test, CV)
F1-Score, Precision, Recall
Mean Squared Error (MSE)
Classification Reports for interpretability

# 🛠 Technologies Used

Language: Python 3
Libraries:
pandas, numpy, scikit-learn, matplotlib
networkx (for graph construction and shortest path)
SMOTE from imblearn for class imbalance handling
Tools: Jupyter Notebook / Google Colab

# 🔮 Future Work

Integration of real-time weather and flight status APIs
Use of ensemble learning or deep learning for enhanced predictions
Incorporation of passenger preferences in route planning
Scalability testing with larger, more diverse datasets

# 📚 References

Kim, J. (2021). A Data-Driven Approach Using Machine Learning for Real-Time Flight Path Optimization
Oza et al. (2017). Flight Delay Prediction Using Weighted Multiple Linear Regression
Bishop, C. M. (2006). Pattern Recognition and Machine Learning
Dataset: GitHub - flights.csv
