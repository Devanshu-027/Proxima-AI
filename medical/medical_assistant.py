"""
Medical Assistant: Rule-Based Diagnosis Tool
=============================================

Description:
    This script provides a simple rule-based medical diagnosis assistant.
    It takes one or more symptoms as input and ranks possible diseases based on
    a predefined symptom-disease mapping. Scores are based on symptom overlaps,
    with higher scores indicating more matches.

Purpose:
    - Demonstrate basic expert system logic for medical diagnosis.
    - Educate on symptom-disease relationships using a knowledge base.
    - Showcase professional Python: argument parsing, data structures, and output formatting.
    - Highlight limitations of rule-based systems vs. AI/ML.

Key Features:
    - Predefined SYMPTOM_MAP: 5python medical_assistant.py --symptoms cough headache fever0+ symptoms mapped to multiple diseases.
    - Scoring: Counts overlaps for ranking.
    - Input handling: Case-insensitive, cleaned, and validated.
    - Ranked output: Sorted by score descending.
    - Extensible: Easy to add more symptoms/diseases or integrate ML (e.g., scikit-learn classifiers).

Prerequisites:
    - Python 3.8+ (no external libraries required).
    - Knowledge base is hardcoded; expand for more accuracy.

Usage:
    python medical_assistant.py --symptoms cough headache fever
    - --symptoms: Space-separated list of symptoms (e.g., cough headache).

Outputs:
    - Console: Ranked list of possible diagnoses with scores.
    - If no matches: Suggestion to expand the map or use ML.

Example:
    python medical_assistant.py --symptoms fever cough fatigue
    # Output: flu (score=3), common cold (score=2), etc.

Limitations:
    - Rule-based: No probabilities or AI; relies on manual mapping.
    - Incomplete: Covers common symptoms but not exhaustive.
    - Not medical advice: Results are speculative; consult a healthcare professional.
    - Accuracy depends on input quality; typos or rare symptoms may miss matches.
    - For production: Integrate real medical datasets or ML models (e.g., trained on patient data).

Author: [Your Name/Username]
License: MIT
"""

import argparse
from collections import defaultdict


# Further expanded rule-based mapping for higher demo accuracy
# Based on general medical knowledge; expand with real data or integrate ML
SYMPTOM_MAP = {
    'fever': ['flu', 'common cold', 'dengue', 'malaria', 'pneumonia', 'covid-19', 'sepsis'],
    'cough': ['common cold', 'bronchitis', 'covid-19', 'pneumonia', 'asthma', 'tuberculosis', 'pertussis'],
    'headache': ['migraine', 'flu', 'stress', 'dehydration', 'sinusitis', 'hypertension', 'meningitis'],
    'rash': ['allergy', 'measles', 'dengue', 'chickenpox', 'eczema', 'scabies', 'psoriasis'],
    'bodyache': ['flu', 'dengue', 'malaria', 'fibromyalgia', 'covid-19', 'rheumatoid arthritis'],
    'sore throat': ['common cold', 'strep throat', 'flu', 'tonsillitis', 'mononucleosis'],
    'fatigue': ['flu', 'anemia', 'depression', 'chronic fatigue syndrome', 'covid-19', 'hypothyroidism'],
    'nausea': ['food poisoning', 'gastritis', 'migraine', 'pregnancy', 'motion sickness', 'gallstones'],
    'vomiting': ['food poisoning', 'gastritis', 'appendicitis', 'migraine', 'pancreatitis'],
    'diarrhea': ['food poisoning', 'gastroenteritis', 'irritable bowel syndrome', 'cholera', 'celiac disease'],
    'shortness of breath': ['asthma', 'pneumonia', 'covid-19', 'heart failure', 'anxiety', 'pulmonary embolism'],
    'chest pain': ['heart attack', 'pneumonia', 'anxiety', 'acid reflux', 'costochondritis', 'angina'],
    'runny nose': ['common cold', 'allergies', 'sinusitis', 'flu', 'rhinitis'],
    'sneezing': ['allergies', 'common cold', 'hay fever', 'flu'],
    'joint pain': ['arthritis', 'lupus', 'gout', 'flu', 'rheumatoid arthritis'],
    'muscle pain': ['flu', 'fibromyalgia', 'injury', 'hypothyroidism', 'myositis'],
    'dizziness': ['dehydration', 'anemia', 'vertigo', 'low blood pressure', 'meniere\'s disease'],
    'loss of appetite': ['flu', 'depression', 'gastritis', 'cancer', 'hepatitis'],
    'sweating': ['flu', 'hyperthyroidism', 'menopause', 'anxiety', 'pheochromocytoma'],
    'chills': ['flu', 'malaria', 'pneumonia', 'sepsis', 'tuberculosis'],
    'itching': ['allergy', 'eczema', 'scabies', 'liver disease', 'psoriasis'],
    'swelling': ['allergy', 'heart failure', 'kidney disease', 'injury', 'lymphedema'],
    'abdominalpain': ['appendicitis', 'gastritis', 'irritable bowel syndrome', 'ulcer', 'diverticulitis'],
    'back pain': ['muscle strain', 'herniated disc', 'kidney stones', 'arthritis', 'sciatica'],
    'frequent urination': ['urinary tract infection', 'diabetes', 'prostate issues', 'interstitial cystitis'],
    'blood in urine': ['urinary tract infection', 'kidney stones', 'bladder cancer', 'glomerulonephritis'],
    'yellow skin': ['jaundice', 'liver disease', 'hepatitis', 'pancreatitis'],
    'weight loss': ['cancer', 'hyperthyroidism', 'depression', 'tuberculosis', 'crohn\'s disease'],
    'night sweats': ['tuberculosis', 'lymphoma', 'menopause', 'brucellosis'],
    'confusion': ['dehydration', 'infection', 'dementia', 'hypoglycemia', 'delirium'],
    'blurred vision': ['diabetes', 'migraine', 'glaucoma', 'cataracts', 'retinopathy'],
    'constipation': ['irritable bowel syndrome', 'hypothyroidism', 'dehydration', 'diverticulitis'],
    'heart palpitations': ['anxiety', 'arrhythmia', 'hyperthyroidism', 'anemia'],
    'tremors': ['parkinson\'s disease', 'anxiety', 'hyperthyroidism', 'essential tremor'],
    'memory loss': ['dementia', 'depression', 'vitamin deficiency', 'hypothyroidism'],
    'insomnia': ['stress', 'anxiety', 'depression', 'hyperthyroidism', 'sleep apnea'],
    'hair loss': ['hypothyroidism', 'alopecia', 'stress', 'anemia', 'chemotherapy'],
    'dry mouth': ['dehydration', 'diabetes', 'sjogren\'s syndrome', 'medication side effect'],
    'bruising': ['vitamin deficiency', 'leukemia', 'liver disease', 'thrombocytopenia'],
    'numbness': ['diabetes', 'multiple sclerosis', 'stroke', 'peripheral neuropathy'],
    'tingling': ['diabetes', 'anxiety', 'multiple sclerosis', 'carpal tunnel syndrome'],
    'difficulty swallowing': ['esophageal cancer', 'achalasia', 'stroke', 'goiter'],
    'hoarseness': ['laryngitis', 'thyroid issues', 'vocal cord paralysis', 'smoking'],
    'ear pain': ['ear infection', 'sinusitis', 'temporomandibular joint disorder'],
    'nosebleed': ['dry air', 'hypertension', 'nasal polyps', 'blood clotting disorder'],
    'eye pain': ['glaucoma', 'migraine', 'conjunctivitis', 'uveitis'],
    'tooth pain': ['tooth decay', 'gum disease', 'sinusitis', 'trigeminal neuralgia'],
    'skin ulcers': ['diabetes', 'vascular disease', 'infection', 'autoimmune disorders'],
    'leg cramps': ['dehydration', 'electrolyte imbalance', 'peripheral artery disease'],
    'cold hands/feet': ['raynaud\'s disease', 'anemia', 'hypothyroidism', 'anxiety'],
    'excessive thirst': ['diabetes', 'dehydration', 'diuretic use', 'hypercalcemia'],
    'frequent infections': ['immunodeficiency', 'diabetes', 'cancer', 'hiv'],
    'mood swings': ['bipolar disorder', 'depression', 'hormonal imbalance', 'thyroid issues']
}


def diagnose(symptoms):
    """
    Diagnoses possible diseases based on input symptoms using the SYMPTOM_MAP.

    Args:
        symptoms (list): List of symptom strings (e.g., ['cough', 'headache']).

    Returns:
        list: Sorted list of tuples (disease, score), ranked by score descending.
    """
    score = defaultdict(int)
    for s in symptoms:
        s = s.strip().lower().replace(' ', '')  # Clean: lowercase, remove spaces (e.g., 'body ache' -> 'bodyache')
        matches = SYMPTOM_MAP.get(s, [])
        for d in matches:
            score[d] += 1
    # Convert to sorted list
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    """
    Main function: Parses arguments, validates input, runs diagnosis, and prints results.
    """
    parser = argparse.ArgumentParser(description="Rule-Based Medical Diagnosis Assistant")
    parser.add_argument('--symptoms', nargs='+', required=True, help="List one or more symptoms (e.g., cough headache)")
    args = parser.parse_args()
    
    if not args.symptoms:
        print("Error: No symptoms provided. Please specify at least one symptom.")
        return
    
    # Validate symptoms (basic check for empty strings)
    cleaned_symptoms = [s for s in args.symptoms if s.strip()]
    if not cleaned_symptoms:
        print("Error: All provided symptoms are empty or invalid.")
        return
    
    results = diagnose(cleaned_symptoms)
    if not results:
        print('No matches found in rule base. Expand SYMPTOM_MAP or add a dataset + ML model.')
        print('Disclaimer: This is for educational purposes only. Consult a healthcare professional for real diagnosis.')
    else:
        print('Possible diagnoses (ranked by score):')
        for diag, sc in results:
            print(f'{diag} (score={sc})')
        print('\nDisclaimer: This is a simple rule-based tool for demonstration. Not a substitute for professional medical advice.')


if __name__ == '__main__':
    main()
