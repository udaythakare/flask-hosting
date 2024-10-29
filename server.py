from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/',methods=['GET'])
def send_page():
    return jsonify({ 'name' : 'uday'})

@app.route('/api/analyze-review', methods=['POST'])
def analyze_review():
    data = request.get_json()
    review = data['review']

    # Sentiment analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(review)
    sentiment_score = sentiment_scores['compound']

    # Generate word cloud
    wordcloud = WordCloud().generate(review)
    word_cloud_data = [{'text': word, 'value': count} for word, count in wordcloud.words_.items()]

    return jsonify({
        'sentiment_score': sentiment_score,
        'word_cloud': word_cloud_data
    })

if __name__ == '__main__':
    app.run(debug=True)