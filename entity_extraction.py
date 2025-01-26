import nltk
import json
import spacy
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd

# Download necessary NLTK resources

# nltk.download('punkt_tab')

# Load the domain knowledge (competitors, features, pricing keywords)
def load_domain_knowledge(file_path='domain_knowledge.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    return text

# Function for dictionary-based entity extraction
def dictionary_based_extraction(text, domain_knowledge):
    extracted_entities = defaultdict(list)
    
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    
    # Check if any tokens match known competitors, features, or pricing keywords
    for token in tokens:
        for category, keywords in domain_knowledge.items():
            if token in keywords:
                extracted_entities[category].append(token)
    
    return extracted_entities

# Function for NER-based extraction using spaCy
def ner_based_extraction(text, nlp):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract named entities using spaCy NER
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    return entities

# Combine dictionary-based and NER-based extractions
def combine_extractions(text, domain_knowledge, nlp):
    dict_entities = dictionary_based_extraction(text, domain_knowledge)
    ner_entities = ner_based_extraction(text, nlp)
    
    # Combine results
    combined_entities = defaultdict(list)
    
    for category, entities in dict_entities.items():
        combined_entities[category].extend(entities)
    
    for category, entities in ner_entities.items():
        combined_entities[category].extend(entities)
    
    return combined_entities

# Function to calculate precision and recall
def calculate_precision_recall(extracted_entities, ground_truth_entities):
    tp = len(set(extracted_entities).intersection(set(ground_truth_entities)))  # True positives
    fp = len(set(extracted_entities).difference(set(ground_truth_entities)))  # False positives
    fn = len(set(ground_truth_entities).difference(set(extracted_entities)))  # False negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall

# Main function to process the dataset and extract entities with evaluation
def extract_entities_from_dataset(dataset_path='preprocessed_calls_dataset.csv', ground_truth_path='ground_truth.csv'):
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Load domain knowledge
    domain_knowledge = load_domain_knowledge()
    
    # Load spaCy NER model
    nlp = spacy.load('en_core_web_sm')
    
    # Load ground truth data (for precision/recall evaluation)
    ground_truth_data = pd.read_csv(ground_truth_path)  # Assume ground truth entities are in this CSV
    
    # Process each text snippet in the dataset
    extracted_data = []
    for index, row in data.iterrows():
        text = row['cleaned_text']
        ground_truth = ground_truth_data.iloc[index]['ground_truth_entities'].split(",")  # Assuming ground truth is comma-separated
        
        # Extract entities
        combined_entities = combine_extractions(text, domain_knowledge, nlp)
        
        # Flatten the combined entities for precision/recall calculation
        extracted_entities = [entity for category in combined_entities.values() for entity in category]
        
        # Calculate precision and recall
        precision, recall = calculate_precision_recall(extracted_entities, ground_truth)
        
        # Store the extracted entities, precision, recall, and original text
        extracted_data.append({
            'text': text,
            'extracted_entities': dict(combined_entities),  # Convert defaultdict to normal dict
            'precision': precision,
            'recall': recall
        })
    
    # Convert results to a DataFrame
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df.to_csv('extracted_entities_with_metrics.csv', index=False)
    print("Entity extraction and evaluation completed and saved to 'extracted_entities_with_metrics.csv'.")

# Run the extraction process
if __name__ == "__main__":
    nltk.download('punkt')
    extract_entities_from_dataset()
