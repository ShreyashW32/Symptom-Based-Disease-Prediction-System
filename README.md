# Symptom-Based-Disease-Prediction-System
This Python-based project predicts diseases from patient symptoms using an ensemble machine learning model. It supports 20 diseases (e.g., Bronchial Asthma, Gastroenteritis, Typhoid) with 43 predefined symptoms (e.g., fever, cough, itching). Key features include:

Model: Combines Logistic Regression, Decision Tree, KNN, Naive Bayes, and Random Forest (50 trees, max depth 10) for robust predictions.
Data: Synthetic dataset of 1000 training and 50 test samples, with disease-specific symptoms defined in data_loader.py.
Input: Interactive symptom selection via numbered list (e.g., "14, 24" for "Fever, Cough"), ensuring valid inputs.
Clinical Rules: Adjusts probabilities (e.g., upweights Bronchial Asthma for "cough," downweights Gastroenteritis without vomiting/diarrhea).
Features:
Model persistence (saves/loads models).
Patient history management with AEST timestamps, overwriting same-name records.
Interactive mode for viewing/adding patients.
Logs uncertain predictions (<50% confidence).
Displays top 3 normalized probabilities.
Files: main.py, data_loader.py, text_processor.py, model files (logistic_regression.py, decision_tree.py, knn.py, naive_bayes.py, random_forest.py), patient_history.json, uncertain_predictions.log.
Purpose: Prototype for predicting diseases like Bronchial Asthma for "fever, cough" accurately, with validation advised by medical professionals.
