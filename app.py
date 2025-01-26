from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import spacy
import json

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load pre-trained models and resources
model_path = "multi_label_classifier.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"
label_binarizer_path = "label_binarizer.pkl"
domain_knowledge_path = "domain_knowledge.json"

with open(model_path, "rb") as model_file, \
     open(vectorizer_path, "rb") as vectorizer_file, \
     open(label_binarizer_path, "rb") as binarizer_file:
    classifier = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)
    label_binarizer = pickle.load(binarizer_file)

with open(domain_knowledge_path, "r") as file:
    domain_knowledge = json.load(file)

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

@app.route('/')
def home():
    """Serve the HTML interface."""
    return render_template("interface.html")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict labels, extract entities, and generate a summary."""
    data = request.get_json()
    if not data or 'snippet' not in data:
        return jsonify({"error": "Invalid input. Please provide a 'snippet' field."}), 400
    
    text_snippet = data['snippet']
    
    # Preprocess the text snippet
    cleaned_text = text_snippet.lower()
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict labels
    predicted_probabilities = classifier.predict_proba(vectorized_text)
    predicted_labels = label_binarizer.inverse_transform((predicted_probabilities > 0.5).astype(int))[0]

    # Extract entities
    extracted_entities = extract_entities(cleaned_text, domain_knowledge, nlp)

    # Generate a summary
    summary = generate_summary(text_snippet, predicted_labels)

    # Return response
    response = {
        "Predicted Labels": predicted_labels,
        "Extracted Entities": extracted_entities,
        "Summary": summary
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
