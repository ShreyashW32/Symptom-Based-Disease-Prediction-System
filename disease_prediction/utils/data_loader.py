import numpy as np


def load_data(num_train=1000, num_test=50):
    # Define disease labels (must match main.py)
    diseases = [
        "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
        "Peptic ulcer diseae", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
        "Hypertension", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
        "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A"
    ]

    # Define disease-specific symptoms
    symptoms_by_disease = {
        "Fungal infection": ["itching", "skin rash", "discoloration"],
        "Allergy": ["sneezing", "itchy eyes", "runny nose", "rash"],
        "GERD": ["heartburn", "acid reflux", "chest pain"],
        "Chronic cholestasis": ["jaundice", "dark urine", "itching"],
        "Drug Reaction": ["rash", "swelling", "fever"],
        "Peptic ulcer diseae": ["abdominal pain", "nausea", "heartburn"],
        "AIDS": ["weight loss", "night sweats", "fever", "fatigue"],
        "Diabetes": ["increased thirst", "frequent urination", "fatigue"],
        "Gastroenteritis": ["vomiting", "diarrhea", "nausea", "fever", "abdominal pain"],
        "Bronchial Asthma": ["cough", "wheezing", "shortness of breath", "chest tightness"],
        "Hypertension": ["headache", "dizziness", "nosebleeds"],
        "Migraine": ["headache", "nausea", "sensitivity to light"],
        "Cervical spondylosis": ["neck pain", "stiffness", "shoulder pain"],
        "Paralysis (brain hemorrhage)": ["weakness", "numbness", "confusion"],
        "Jaundice": ["yellow skin", "dark urine", "fatigue"],
        "Malaria": ["fever", "chills", "sweating", "headache"],
        "Chicken pox": ["rash", "fever", "itching"],
        "Dengue": ["high fever", "headache", "muscle pain", "rash"],
        "Typhoid": ["prolonged fever", "abdominal pain", "fatigue", "cough"],
        "hepatitis A": ["jaundice", "nausea", "fatigue", "abdominal pain"]
    }

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for i, disease in enumerate(diseases):
        available_symptoms = symptoms_by_disease[disease]
        max_sample_size = len(available_symptoms)
        # Generate training samples
        for _ in range(num_train // len(diseases)):
            # Ensure sample size <= available symptoms
            sample_size = min(np.random.randint(2, 5), max_sample_size)
            symptoms = np.random.choice(
                available_symptoms, size=sample_size, replace=False)
            text = ", ".join(symptoms)
            train_texts.append(text)
            train_labels.append(i)
        # Generate test samples
        for _ in range(num_test // len(diseases)):
            # Ensure sample size <= available symptoms
            sample_size = min(np.random.randint(2, 5), max_sample_size)
            symptoms = np.random.choice(
                available_symptoms, size=sample_size, replace=False)
            text = ", ".join(symptoms)
            test_texts.append(text)
            test_labels.append(i)

    return train_texts, train_labels, test_texts, test_labels
