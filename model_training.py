import pandas as pd
import pickle
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import spacy

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load domain knowledge
domain_knowledge = {
    "competitors": ["CompetitorX", "CompetitorY", "CompetitorZ"],
    "features": ["analytics", "AI engine", "data pipeline"],
    "pricing_keywords": ["discount", "renewal cost", "budget", "pricing model"]
}

def extract_entities(text, domain_knowledge, nlp):
    """Extract entities using dictionary-based and NER-based methods."""
    extracted_entities = defaultdict(list)
    tokens = word_tokenize(text.lower())
    for category, keywords in domain_knowledge.items():
        for token in tokens:
            if token in keywords:
                extracted_entities[category].append(token)
    doc = nlp(text)
    for ent in doc.ents:
        extracted_entities[ent.label_].append(ent.text)
    combined_entities = {key: list(set(value)) for key, value in extracted_entities.items()}
    return combined_entities

def generate_summary(text, labels):
    """Generate a summary based on predicted labels."""
    summary = f"The text snippet discusses {', '.join(labels)}."
    return summary

def plot_confusion_matrix(y_test, y_pred, class_names):
    """Plot a confusion matrix for multi-label classification."""
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def train_and_save_model(data_path):
    """Train a multi-label classifier and save the model, vectorizer, and label binarizer."""
    data = pd.read_csv(data_path)
    vectorizer = TfidfVectorizer(max_features=5000)
    mlb = MultiLabelBinarizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = mlb.fit_transform(data['labels'].str.split(", "))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
    param_grid = {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [10, 20, None],
        "estimator__min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(model, param_grid, scoring="f1_macro", cv=3, verbose=1, n_jobs=-1)
    logging.info("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    pickle.dump(best_model, open("multi_label_classifier.pkl", "wb"))
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    pickle.dump(mlb, open("label_binarizer.pkl", "wb"))
    logging.info("Model saved successfully.")
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))
    logging.info("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, mlb.classes_)

def predict_labels_and_extract_entities(text_snippet):
    """Predict labels, extract entities, and generate a summary."""
    with open("multi_label_classifier.pkl", "rb") as model_file, \
         open("tfidf_vectorizer.pkl", "rb") as vectorizer_file, \
         open("label_binarizer.pkl", "rb") as binarizer_file:
        classifier = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
        label_binarizer = pickle.load(binarizer_file)
    cleaned_text = text_snippet.lower()
    vectorized_text = vectorizer.transform([cleaned_text])
    predicted_probabilities = classifier.predict_proba(vectorized_text)
    predicted_labels = label_binarizer.inverse_transform((predicted_probabilities > 0.5).astype(int))[0]
    extracted_entities = extract_entities(cleaned_text, domain_knowledge, nlp)
    summary = generate_summary(text_snippet, predicted_labels)
    return {
        "Predicted Labels": predicted_labels,
        "Extracted Entities": extracted_entities,
        "Summary": summary
    }

if __name__ == "__main__":
    # Step 1: Train and save the model
    data_path = "preprocessed_calls_dataset.csv"
    train_and_save_model(data_path)

    # Step 2: Predict and extract entities for an example
    example_text = "We love the analytics, but CompetitorX has a cheaper subscription."
    result = predict_labels_and_extract_entities(example_text)

    # Print results
    print("\nResults:")
    print(f"Predicted Labels: {result['Predicted Labels']}")
    print(f"Extracted Entities: {result['Extracted Entities']}")
    print(f"Summary: {result['Summary']}")
