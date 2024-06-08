from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import pandas as pd
import requests
import threading
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import io
import base64
import csv

app = Flask(__name__)

# Global variable to track progress
progress = 0

def convert_to_review_url(base_url):
    parts = base_url.split('/p/')
    if len(parts) != 2:
        raise ValueError("URL format is not as expected.")
    review_page_url = parts[0] + '/product-reviews/' + parts[1]
    review_page_url = review_page_url.split('?')[0] + '?' + '&'.join(
        [param for param in base_url.split('?')[1].split('&') if param.startswith('pid=') or param.startswith('lid=') or param.startswith('marketplace=')]
    )
    return review_page_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global progress
    progress = 0  # Reset progress

    url = request.json['url']
    thread = threading.Thread(target=scrape_and_clean_data, args=(url,))
    thread.start()
    return jsonify({'message': 'Web scraping started.'})

@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})

@app.route('/preferences')
def preferences():
    return render_template('preferences.html')

@app.route('/perform_analysis', methods=['POST'])
def perform_analysis():
    criteria = request.form.get('criteria', '').lower()
    num_reviews = int(request.form.get('num_reviews', 10))
    criteria_list = [c.strip() for c in criteria.split(',')]

    relevant_comments = search_comments('product_reviews.csv', criteria_list, num_reviews)
    
    if not relevant_comments:
        return render_template('analysis.html', message=f"No reviews found for criteria: {criteria}")

    sentiments = analyze_sentiments(relevant_comments)

    pie_chart_url = generate_pie_chart(sentiments)

    overall_sentiment = "Worth Buying" if sentiments['positive'] > (sentiments['negative'] + sentiments['neutral']) else "Not Worth Buying"

    return render_template('analysis.html', sentiments=sentiments, comments=relevant_comments, pie_chart_url=pie_chart_url, overall_sentiment=overall_sentiment)

def scrape_and_clean_data(base_url):
    global progress
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept-Language': 'en-us,en;q=0.5'
    }

    customer_names = []
    review_title = []
    ratings = []
    comments = []

    review_url = convert_to_review_url(base_url)
    total_pages = 50  # Set to 50 pages to get approximately 500 reviews (10 reviews per page)
    
    for i in range(1, total_pages + 1):
        current_url = f"{review_url}&page={i}"
        page = requests.get(current_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')

        names = soup.find_all('p', class_='_2NsDsF AwS1CA')
        for name in names:
            customer_names.append(name.get_text())

        title = soup.find_all('p', class_='z9E0IG')
        for t in title:
            review_title.append(t.get_text())

        rat = soup.find_all('div', class_='XQDdHH Ga3i8K')
        for r in rat:
            rating = r.get_text()
            if rating:
                ratings.append(rating)
            else:
                ratings.append('0')

        cmt = soup.find_all('div', class_='ZmyHeo')
        for c in cmt:
            comment_text = c.div.div.get_text(strip=True)
            comments.append(comment_text)

        progress = int((i / total_pages) * 100)  # Update progress based on total pages processed

    min_length = min(len(customer_names), len(review_title), len(ratings), len(comments))
    customer_names = customer_names[:min_length]
    review_title = review_title[:min_length]
    ratings = ratings[:min_length]
    comments = comments[:min_length]

    data = {
        'Customer Name': customer_names,
        'Review Title': review_title,
        'Rating': ratings,
        'Comment': comments
    }

    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)

    # Data Cleaning Steps
    def clean_text(text):
        text = re.sub(r"(http\S+|www\S+)", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    df['clean_text'] = df['Comment'].apply(clean_text)

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = tokenizer.tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    df['Cleaned_text'] = df['clean_text'].apply(preprocess_text)

    selected_columns = ['Customer Name', 'Review Title', 'Rating', 'Comment', 'Cleaned_text']
    cleaned_df = df[selected_columns]
    cleaned_df.to_csv('product_reviews.csv', index=False)
    
    progress = 100  # Ensure progress is set to 100% when done

def analyze_sentiment(comment):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(comment)
    return sentiment_scores

def search_comments(csv_file, criteria, num_reviews):
    df = pd.read_csv(csv_file)
    df['Cleaned_text'] = df['Cleaned_text'].fillna('')
    relevant_comments = []
    for _, row in df.iterrows():
        comment = row['Cleaned_text'].lower()
        if any(word in comment for word in criteria):
            relevant_comments.append(row['Comment'])
            if len(relevant_comments) == num_reviews:
                break  # Stop after reaching the specified number of reviews
    return relevant_comments

def analyze_sentiments(comments):
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    for comment in comments:
        sentiment_scores = analyze_sentiment(comment)
        if sentiment_scores['compound'] >= 0.05:
            sentiments['positive'] += 1
        elif sentiment_scores['compound'] <= -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

def generate_pie_chart(sentiments):
    labels = sentiments.keys()
    sizes = sentiments.values()
    explode = (0.1, 0, 0)  # Explode the 1st slice (positive sentiment)

    plt.figure(figsize=(4, 4))  # Adjust the size of the pie chart
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title('Sentiment Analysis of Reviews based on User-Specific Choice')

    # Save the plot to a string buffer and encode it in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pie_chart_url = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return pie_chart_url

if __name__ == '__main__':
    app.run(debug=True)
