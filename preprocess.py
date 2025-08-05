# preprocess.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ✅ Function to clean raw text (remove URLs, special characters, etc.)
def clean_text(text):
    # Remove any URL-like patterns
    text = re.sub(r"http\S+", "", text)
    # Remove non-alphabet characters (e.g., numbers, punctuation)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Convert all text to lowercase
    return text.lower()

# ✅ Load and prepare both Fake and Real news datasets
def load_and_prepare_data():
    # Load CSV files
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    # Assign labels: Fake = 0, Real = 1
    df_fake['label'] = 0
    df_true['label'] = 1

    # Combine both datasets into one
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Merge title and text into a single column for analysis
    df['text'] = df['title'] + " " + df['text']

    # Clean the text data
    df['text'] = df['text'].apply(clean_text)

    # Return only the text and label columns
    return df[['text', 'label']]

# ✅ Vectorize the text data using TF-IDF
def vectorize_texts(X_train, fit=True):
    """
    If `fit=True`, a new vectorizer is trained and saved to disk.
    If `fit=False`, the saved vectorizer is loaded and used to transform new data.
    """
    if fit:
        # Create and train a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
        X_vec = vectorizer.fit_transform(X_train)
        # Save the trained vectorizer for reuse
        joblib.dump(vectorizer, 'vectorizer.joblib')
    else:
        # Load the pre-trained vectorizer
        vectorizer = joblib.load('vectorizer.joblib')
        # Use it to transform the input text
        X_vec = vectorizer.transform(X_train)

    # Return the transformed text as sparse matrix
    return X_vec
