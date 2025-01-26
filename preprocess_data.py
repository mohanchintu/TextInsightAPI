import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import logging

# Download necessary NLTK data
nltk.download('stopwords')

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def clean_text(text, stop_words):
    """Clean the input text by removing non-alphabet characters and stopwords."""
    if pd.isnull(text):  # Handle missing text
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabets and spaces
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

def preprocess_dataset(data):
    """Preprocess the dataset by cleaning the 'text_snippet' column."""
    stop_words = set(stopwords.words('english'))
    data['cleaned_text'] = data['text_snippet'].apply(lambda x: clean_text(x, stop_words))
    logging.info("Dataset preprocessing complete.")
    return data

def save_preprocessed_data(data, file_path):
    """Save the preprocessed dataset to a CSV file."""
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Preprocessed dataset saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving preprocessed dataset: {e}")
        raise

if __name__ == "__main__":
    # Define input and output files
    input_file = "calls_dataset.csv"  # Input dataset
    output_file = "preprocessed_calls_dataset.csv"  # Preprocessed dataset

    logging.info("Starting preprocessing pipeline...")

    try:
        # Load dataset
        dataset = load_dataset(input_file)

        # Preprocess data
        dataset = preprocess_dataset(dataset)

        # Save preprocessed dataset
        save_preprocessed_data(dataset, output_file)

        logging.info("Preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
