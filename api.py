from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flasgger import Swagger
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import logging
import warnings
from helpers import vn_processing as xt

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
swagger = Swagger(app)  # Initialize Swagger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained models and vectorizer
with open('models/xgboots_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to preprocess text and make predictions
def predict_sentiment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = model.predict(X_test)
    df['Label'] = y_pred
    df['Label'] = df['Label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    df = df[['Comment', 'Label']]
    return df.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json  # Get JSON data from request
        text = content.get('text')  # Access 'text' key from JSON data
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        result = predict_sentiment([text])
        return jsonify(result)
    except KeyError:
        return jsonify({'error': 'Missing required parameter'}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/apidocs')
def apidocs():
    return render_template('swaggerui.html')

# Define API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Sentiment analysis API
    ---
    parameters:
      - name: comment
        in: body
        type: string
        required: true
        description: The comment text for sentiment analysis
    responses:
      200:
        description: Sentiment analysis result
    """
    comments = request.json.get('comment')
    predictions = predict_sentiment(comments)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
