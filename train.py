# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from preprocess import load_and_prepare_data, vectorize_texts

# Load and prepare
df = load_and_prepare_data()

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Stratified split: 10% labeled from both classes
df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1]

labeled_fake = df_fake.sample(frac=0.1, random_state=42)
labeled_real = df_real.sample(frac=0.1, random_state=42)

X_labeled = pd.concat([labeled_fake['text'], labeled_real['text']])
y_labeled = pd.concat([labeled_fake['label'], labeled_real['label']])

# Unlabeled = the rest
df_remaining = df.drop(pd.concat([labeled_fake, labeled_real]).index)
X_unlabeled = df_remaining['text']
y_unlabeled = [-1] * len(X_unlabeled)

# Combine labeled + unlabeled
X_all = pd.concat([X_labeled, X_unlabeled])
y_all = np.array(list(y_labeled) + y_unlabeled)

# Vectorize
X_vec = vectorize_texts(X_all, fit=True)

# Semi-supervised training
base_model = LogisticRegression(max_iter=1000)
self_training_model = SelfTrainingClassifier(
    base_model,
    criterion='k_best',
    k_best=500,          # limit how many pseudo-labels to accept per iteration
    threshold=0.8        # only use predictions with high confidence
)
self_training_model.fit(X_vec, y_all)

# Evaluate on held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

X_test_vec = vectorize_texts(X_test, fit=False)
y_pred = self_training_model.predict(X_test_vec)

print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(self_training_model, 'model.joblib')
