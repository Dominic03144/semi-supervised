# app.py
from flask import Flask, request, jsonify, render_template
import joblib
from preprocess import clean_text, vectorize_texts
import os

app = Flask(__name__)
model = joblib.load('model.joblib')

# --- Root endpoint (status check) ---
@app.route('/', methods=['GET'])
def home():
    return "📰 Fake News Detection API is running!"

# --- JSON API endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = clean_text(data['text'])
    vector = vectorize_texts([text], fit=False)
    prediction = model.predict(vector)[0]
    result = 'REAL' if prediction == 1 else 'FAKE'
    return jsonify({'prediction': result})

# --- HTML UI form route ---
@app.route('/form', methods=['GET'])
def form():
    return render_template('index.html')

# --- Handle form submission ---
@app.route('/predict-form', methods=['POST'])
def predict_form():
    user_text = request.form['text']
    cleaned = clean_text(user_text)
    vector = vectorize_texts([cleaned], fit=False)
    prediction = model.predict(vector)[0]
    result = 'REAL 🟢' if prediction == 1 else 'FAKE 🔴'
    return render_template('index.html', prediction=result, original=user_text)

if __name__ == '__main__':
    app.run(debug=True)
