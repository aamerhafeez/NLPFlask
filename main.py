from flask import Flask, request, jsonify
import stanza
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download Stanza model (required only the first time)
#stanza.download('en')
stanza.download('en', package='wsj')

# Load Stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,sentiment', batch_size=8, use_gpu=False)


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    # Get comments from POST request
    data = request.json
    comments = data['comments']
    results = []

    # Analyze each comment using Stanza
    for comment in comments:
        doc = nlp(comment)
        sentiment_score = 0
        for sentence in doc.sentences:
            sentiment_score = sentence.sentiment  # 0=negative, 1=neutral, 2=positive
        results.append({'comment': comment, 'sentiment': sentiment_score})

    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Make Flask app accessible over your office network
