import numpy as np
import datetime
import json
import os
import pickle
import pytz
from utils.data_loader import load_data
from utils.text_processor import build_vocabulary, texts_to_vectors
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTree
from models.knn import KNN
from models.naive_bayes import MultinomialNB
from models.random_forest import RandomForest

# Define disease labels
diseases = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
    "Peptic ulcer diseae", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
    "Hypertension", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
    "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A"
]

# Define symptom list
symptoms = [
    "Itching", "Skin rash", "Discoloration", "Sneezing", "Itchy eyes", "Runny nose",
    "Rash", "Heartburn", "Acid reflux", "Chest pain", "Jaundice", "Dark urine",
    "Swelling", "Fever", "Abdominal pain", "Nausea", "Weight loss", "Night sweats",
    "Fatigue", "Increased thirst", "Frequent urination", "Vomiting", "Diarrhea",
    "Cough", "Wheezing", "Shortness of breath", "Chest tightness", "Headache",
    "Dizziness", "Nosebleeds", "Sensitivity to light", "Neck pain", "Stiffness",
    "Shoulder pain", "Weakness", "Numbness", "Confusion", "Yellow skin", "Chills",
    "Sweating", "High fever", "Muscle pain", "Prolonged fever"
]

patient_history_file = 'patient_history.json'
saved_models_dir = 'saved_models'
uncertain_log_file = 'uncertain_predictions.log'
model_files = {
    'lr': os.path.join(saved_models_dir, 'logistic_regression.pkl'),
    'dt': os.path.join(saved_models_dir, 'decision_tree.pkl'),
    'knn': os.path.join(saved_models_dir, 'knn.pkl'),
    'nb': os.path.join(saved_models_dir, 'naive_bayes.pkl'),
    'rf': os.path.join(saved_models_dir, 'random_forest.pkl'),
    'word_to_idx': os.path.join(saved_models_dir, 'word_to_idx.pkl')
}


def load_patient_history():
    try:
        if os.path.exists(patient_history_file):
            with open(patient_history_file, 'r') as f:
                return json.load(f)
        return []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading patient history: {e}")
        return []


def save_patient_history(history):
    try:
        with open(patient_history_file, 'w') as f:
            json.dump(history, f, indent=4)
    except IOError as e:
        print(f"Error saving patient history: {e}")


def log_uncertain_prediction(name, symptoms, predicted_disease, probs):
    aest = pytz.timezone('Australia/Sydney')
    timestamp = datetime.datetime.now(aest).strftime("%Y-%m-%d %H:%M:%S %Z")
    top_prob = max(probs) * 100
    if top_prob < 50:
        log_entry = {
            'timestamp': timestamp,
            'name': name,
            'symptoms': symptoms,
            'predicted_disease': predicted_disease,
            'top_probability': top_prob,
            'top_3_diseases': {diseases[i]: float(probs[i] * 100) for i in np.argsort(probs)[-3:][::-1]}
        }
        try:
            with open(uncertain_log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except IOError as e:
            print(f"Error logging uncertain prediction: {e}")


def add_to_history(history, name, age, gender, text, predicted_disease, probabilities, actual_disease=None):
    # Remove existing records with the same name (case-insensitive)
    history[:] = [record for record in history if record['name'].lower()
                  != name.lower()]
    # Use AEST timezone for timestamp
    aest = pytz.timezone('Australia/Sydney')
    timestamp = datetime.datetime.now(aest).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = {
        'name': name,
        'age': age,
        'gender': gender,
        'symptoms': text,
        'predicted_disease': predicted_disease,
        'probabilities': {diseases[i]: float(prob * 100) for i, prob in enumerate(probabilities)},
        'actual_disease': actual_disease,
        'timestamp': timestamp
    }
    history.append(entry)


def view_patient_history_by_name(history):
    name = input("Enter patient name to view history (or 'cancel' to skip): ")
    if name.lower() == 'cancel':
        return
    matching_records = [
        record for record in history if record['name'].lower() == name.lower()]
    if not matching_records:
        print(f"No records found for patient: {name}")
        return
    print(f"\nRecord for {name}:")
    record = matching_records[0]  # Only one record per name due to overwriting
    print(f"Age: {record['age']}")
    print(f"Gender: {record['gender']}")
    print(f"Symptoms: {record['symptoms']}")
    print(f"Predicted Disease: {record['predicted_disease']}")
    print(
        f"Actual Disease: {record['actual_disease'] if record['actual_disease'] else 'Not Diagnosed'}")
    print(f"Timestamp: {record['timestamp']}")
    # Display top 3 probabilities normalized to 100%
    probs = [record['probabilities'][disease] /
             100 for disease in diseases]  # Convert back to 0-1 scale
    top_indices = np.argsort(probs)[-3:][::-1]
    top_diseases = [diseases[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    total_top_prob = sum(top_probs)
    if total_top_prob > 0:
        top_probs = [prob / total_top_prob * 100 for prob in top_probs]
    else:
        top_probs = [0.0] * 3
    print("Top 3 Disease Probabilities (%):")
    for disease, prob in zip(top_diseases, top_probs):
        print(f"{disease}: {prob:.2f}%")
    print()


def predict_new_patient(text, word_to_idx, models, rf):
    vector = texts_to_vectors([text], word_to_idx)[0:1]  # Shape: (1, D)
    lr_prob = models['lr'].predict_proba(vector)[0]
    dt_prob = models['dt'].predict_proba(vector)[0]
    knn_prob = models['knn'].predict_proba(vector)[0]
    nb_prob = models['nb'].predict_proba(vector)[0]
    meta_feature = np.hstack([lr_prob, dt_prob, knn_prob, nb_prob])[
        np.newaxis, :]  # Shape: (1, D)
    final_probs = np.mean([tree.predict_proba(meta_feature)
                          for tree in rf.trees], axis=0)[0]
    # Clinical adjustments
    if "cough" in text.lower() and "vomiting" not in text.lower() and "diarrhea" not in text.lower():
        # Downweight Gastroenteritis
        final_probs[diseases.index("Gastroenteritis")] *= 0.1
        final_probs[diseases.index("Bronchial Asthma")
                    ] *= 1.5  # Upweight Asthma
    if "cough" in text.lower() and "weight loss" not in text.lower():
        final_probs[diseases.index("AIDS")] *= 0.1  # Downweight AIDS
    final_probs /= final_probs.sum()  # Renormalize
    predicted_idx = np.argmax(final_probs)
    return diseases[predicted_idx], final_probs


def save_models(models, rf, word_to_idx):
    os.makedirs(saved_models_dir, exist_ok=True)
    try:
        for model_name, model_file in model_files.items():
            with open(model_file, 'wb') as f:
                if model_name in models:
                    pickle.dump(models[model_name], f)
                elif model_name == 'rf':
                    pickle.dump(rf, f)
                elif model_name == 'word_to_idx':
                    pickle.dump(word_to_idx, f)
    except IOError as e:
        print(f"Error saving models: {e}")


def load_models():
    try:
        models = {}
        rf = None
        word_to_idx = None
        for model_name, model_file in model_files.items():
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    if model_name in ['lr', 'dt', 'knn', 'nb']:
                        models[model_name] = pickle.load(f)
                    elif model_name == 'rf':
                        rf = pickle.load(f)
                    elif model_name == 'word_to_idx':
                        word_to_idx = pickle.load(f)
            else:
                return None, None, None
        return models, rf, word_to_idx
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading models: {e}")
        return None, None, None


def get_symptoms_input():
    print("\nSelect symptoms by entering numbers (e.g., '1, 2, 3' or '1' for one symptom).")
    print("Enter 'done' when finished selecting symptoms.")
    for i, symptom in enumerate(symptoms, 1):
        print(f"{i}. {symptom}")

    selected_symptoms = []
    while True:
        user_input = input(
            "\nEnter symptom number(s) or 'done': ").strip().lower()
        if user_input == 'done':
            if not selected_symptoms:
                print("No symptoms selected. Please select at least one symptom.")
                continue
            break
        try:
            # Parse input (e.g., "1, 2, 3" or "1")
            numbers = [int(num.strip()) for num in user_input.replace(
                ',', ' ').split() if num.strip().isdigit()]
            if not numbers:
                print("Invalid input. Please enter numbers or 'done'.")
                continue
            for num in numbers:
                if 1 <= num <= len(symptoms):
                    symptom = symptoms[num - 1]
                    if symptom not in selected_symptoms:
                        selected_symptoms.append(symptom)
                    else:
                        print(f"Symptom '{symptom}' already selected.")
                else:
                    print(
                        f"Number {num} is out of range. Choose between 1 and {len(symptoms)}.")
        except ValueError:
            print("Invalid input. Please enter numbers or 'done'.")

    return ", ".join(selected_symptoms)


def main():
    # Load patient history
    patient_history = load_patient_history()

    # Check if saved models exist
    models, rf, word_to_idx = load_models()
    if models and rf and word_to_idx:
        print("Loaded saved models. Skipping training.")
    else:
        print("No saved models found. Training models...")
        # Load and prepare data
        train_texts, train_labels, test_texts, test_labels = load_data(
            num_train=1000, num_test=50)
        vocabulary, word_to_idx = build_vocabulary(train_texts)
        train_features = texts_to_vectors(train_texts, word_to_idx)
        test_features = texts_to_vectors(test_texts, word_to_idx)
        num_classes = len(np.unique(train_labels))

        # Initialize models with tuned hyperparameters
        lr = LogisticRegression(num_classes=num_classes,
                                learning_rate=0.05, num_iterations=2000)
        dt = DecisionTree(max_depth=10, num_classes=num_classes)
        knn = KNN(k=5, num_classes=num_classes)
        nb = MultinomialNB(alpha=0.5)
        models = {'lr': lr, 'dt': dt, 'knn': knn, 'nb': nb}

        # Train base models
        print("Training Logistic Regression...")
        lr.fit(train_features, train_labels)
        print("Training Decision Tree...")
        dt.fit(train_features, train_labels)
        print("Training KNN...")
        knn.fit(train_features, train_labels)
        print("Training Naive Bayes...")
        nb.fit(train_features, train_labels)

        # Prepare meta-features for Random Forest
        lr_probs_train = lr.predict_proba(train_features)
        dt_probs_train = dt.predict_proba(train_features)
        knn_probs_train = knn.predict_proba(train_features)
        nb_probs_train = nb.predict_proba(train_features)
        meta_features_train = np.hstack(
            [lr_probs_train, dt_probs_train, knn_probs_train, nb_probs_train])

        # Train Random Forest
        print("Training Random Forest...")
        rf = RandomForest(n_trees=50, max_depth=10, num_classes=num_classes)
        rf.fit(meta_features_train, train_labels)

        # Save models
        save_models(models, rf, word_to_idx)

        # Test predictions and record history
        print("\nPredicting and saving test patient histories:")
        for text, true_label in zip(test_texts, test_labels):
            pred, probs = predict_new_patient(text, word_to_idx, models, rf)
            top_indices = np.argsort(probs)[-3:][::-1]
            top_diseases = [diseases[i] for i in top_indices]
            top_probs = [probs[i] for i in top_indices]
            total_top_prob = sum(top_probs)
            if total_top_prob > 0:
                top_probs = [prob / total_top_prob * 100 for prob in top_probs]
            else:
                top_probs = [0.0] * 3
            add_to_history(patient_history, f"Test Patient", 0,
                           "Unknown", text, pred, probs, diseases[true_label])
            print(
                f"Text: {text} | Predicted: {pred} | Actual: {diseases[true_label]}")
            print("Top 3 Disease Probabilities (%):")
            for disease, prob in zip(top_diseases, top_probs):
                print(f"{disease}: {prob:.2f}%")
            print()

    # View patient history by name (initial prompt)
    view_patient_history_by_name(patient_history)

    # Interactive mode with history viewing loop
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        # History viewing loop
        while True:
            action = input(
                "Enter 'view' to see patient history or 'add' to add a new patient (or 'exit' to quit): ")
            if action.lower() == 'exit':
                # Save patient history before exiting
                save_patient_history(patient_history)
                print("Patient history saved to 'patient_history.json'.")
                return
            elif action.lower() == 'view':
                view_patient_history_by_name(patient_history)
            elif action.lower() == 'add':
                break
            else:
                print("Invalid input. Please enter 'view', 'add', or 'exit'.")

        # Add new patient
        try:
            name = input("Enter patient name: ")
            if name.lower() == 'exit':
                save_patient_history(patient_history)
                print("Patient history saved to 'patient_history.json'.")
                return
            age = int(input("Enter patient age: "))
            gender = input("Enter patient gender (M/F/Other): ")
            symptoms = get_symptoms_input()
            actual_disease = input(
                "Enter actual disease (if known, or press Enter to skip): ").strip() or None
            # Validate actual_disease
            if actual_disease and actual_disease not in diseases:
                print(
                    f"Warning: '{actual_disease}' is not in the disease list.")
                print("Valid diseases:", ", ".join(diseases))
                retry = input(
                    "Enter a valid disease or 'skip' to set as Not Diagnosed: ")
                if retry.lower() == 'skip':
                    actual_disease = None
                elif retry in diseases:
                    actual_disease = retry
                else:
                    print(
                        f"Invalid input. Saving '{actual_disease}' as provided.")
            pred, probs = predict_new_patient(
                symptoms, word_to_idx, models, rf)
            log_uncertain_prediction(
                name, symptoms, pred, probs)  # Log if top probability < 50%
            top_indices = np.argsort(probs)[-3:][::-1]
            top_diseases = [diseases[i] for i in top_indices]
            top_probs = [probs[i] for i in top_indices]
            total_top_prob = sum(top_probs)
            if total_top_prob > 0:
                top_probs = [prob / total_top_prob * 100 for prob in top_probs]
            else:
                top_probs = [0.0] * 3
            print(f"\nPredicted Disease: {pred}")
            print("Top 3 Disease Probabilities (%):")
            for disease, prob in zip(top_diseases, top_probs):
                print(f"{disease}: {prob:.2f}%")
            add_to_history(patient_history, name, age, gender,
                           symptoms, pred, probs, actual_disease)
            print()
        except ValueError:
            print("Invalid age input. Please enter a number.")
            continue


if __name__ == "__main__":
    main()
