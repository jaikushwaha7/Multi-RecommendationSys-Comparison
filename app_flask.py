import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from flask import Flask, request, render_template
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import chardet  # For detecting file encoding

app = Flask(__name__)


# Check file encoding and special characters
def load_data_with_encoding_check(filepath):
    """Load data with encoding detection and non-UTF-8 character handling"""
    # Detect file encoding
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())

    print(f"Detected encoding: {result['encoding']} with confidence {result['confidence']}")

    try:
        # Try reading with detected encoding
        df = pd.read_csv(filepath, encoding=result['encoding'])
        print("Successfully loaded with detected encoding")
    except:
        try:
            # Fallback to UTF-8 with error handling
            df = pd.read_csv(filepath, encoding='utf-8', errors='replace')
            print("Used UTF-8 with replacement for non-UTF-8 characters")
        except:
            # Final fallback to latin1
            df = pd.read_csv(filepath, encoding='latin1')
            print("Used latin1 encoding as final fallback")

    return df
# Load and preprocess data
def load_data():
    data = load_data_with_encoding_check('data/car_reviews_full.csv')

    # Debug: Print column names
    print("Column names in CSV:", data.columns.tolist())
    # Normalize column names (strip whitespace, convert to lowercase)
    columns = data.columns.str.strip().str.lower()
    # Verify required columns
    required_columns = ['brand', 'car model', 'rating', 'review text', 'comfort', 'interior', 'performance', 'value',
                        'exterior', 'reliability', 'reviewer']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Column '{col}' not found in CSV. Available columns: {data.columns.tolist()}")

    # Clean data
    columns_to_process = ['Comfort', 'Interior', 'Performance', 'Value', 'Exterior', 'Reliability']
    # Replace '—' and other non-numeric values
    for col in columns_to_process:
        data[columns_to_process] = data[columns_to_process].replace('—', 0)
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # filtering NA
    data_cleaned = data[['Brand','Car Model','Reviewer', 'Rating','Comfort', 'Interior', 'Performance', 'Value', 'Exterior', 'Reliability','Used For']].fillna(0)
    data_cleaned['Review Text'] = data['Review Text'].astype(str)
    # Normalize ratings
    rating_columns = ['Rating','Comfort', 'Interior', 'Performance', 'Value', 'Exterior', 'Reliability']
    for col in rating_columns:
        data_cleaned[col] = data_cleaned[col].astype(float)
    # Sentiment analysis on review text

    data_cleaned['Sentiment'] = data_cleaned['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data_cleaned


data = load_data()


# Rule-Based Recommendation
# def rule_based_recommendation(preferences):
#     # Example rule: Recommend luxury sedans if user prefers high comfort and interior
#     filtered = data[
#         (data['Comfort'] >= 4.5) &
#         (data['Interior'] >= 4.5) &
#         (data['Brand'].str.lower().isin(['lincoln', 'volvo']))
#         ]
#     if preferences.get('brand'):
#         filtered = filtered[filtered['Brand'].str.lower() == preferences['brand'].lower()]
#     return filtered[['Brand', 'Car Model', 'Rating']].head(3).to_dict('records')
import pandas as pd
from textblob import TextBlob


def rule_based_recommendation(preferences):
    """Provides car recommendations based on user preferences and review data"""
    recommendations = data.copy()

    # Add sentiment analysis if not already present
    if 'sentiment' not in recommendations.columns:
        recommendations['sentiment'] = recommendations['Review Text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )

    # Apply filters based on preferences
    if 'brand' in preferences and preferences['brand']:
        recommendations = recommendations[
            recommendations['Brand'].str.lower() == preferences['brand'].lower()
            ]

    if 'primary_use' in preferences and preferences['primary_use']:
        use_case = preferences['primary_use'].lower()

        if use_case == 'luxury':
            recommendations = recommendations[
                (recommendations['Comfort'] >= 4.0) &
                (recommendations['Review Text'].str.contains('heated|comfortable|luxury', case=False, na=False))
                ].sort_values(['Comfort', 'Rating'], ascending=[False, False])

        elif use_case == 'family':
            recommendations = recommendations[
                (recommendations['Reliability'] >= 4.0) &
                (recommendations['Review Text'].str.contains('family|kids|safety', case=False, na=False))
                ].sort_values(['Reliability', 'Rating'], ascending=[False, False])

        elif use_case == 'commuter':
            recommendations = recommendations[
                (recommendations['Value'] >= 4.0) &
                (recommendations['Used For'] == 'Commuting') &
                (recommendations['Review Text'].str.contains('mpg|mileage|economy', case=False, na=False))
                ].sort_values(['Value', 'Rating'], ascending=[False, False])

        elif use_case == 'performance':
            recommendations = recommendations[
                (recommendations['Performance'] >= 4.0) &
                (recommendations['Review Text'].str.contains('handling|performance|quick|power', case=False, na=False))
                ].sort_values(['Performance', 'Rating'], ascending=[False, False])

    if 'min_reliability' in preferences:
        recommendations = recommendations[
            recommendations['Reliability'] >= float(preferences['min_reliability'])
            ]

    if 'min_comfort' in preferences:
        recommendations = recommendations[
            recommendations['Comfort'] >= float(preferences['min_comfort'])
            ]

    # Convert to list of dictionaries and return top 5
    return recommendations.head(5).to_dict('records')

# Content-Based Recommendation
def content_based_recommendation(preferences):
    """Enhanced content-based recommendation using TF-IDF and multiple features"""
    try:
        # Create a combined text feature
        data['combined_features'] = (
                data['Review Text'] + ' ' +
                data['Brand'] + ' ' +
                data['Car Model'] + ' ' +
                data['Used For'].fillna('')
        )

        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(data['combined_features'])

        # If specific car model is provided
        if preferences.get('car_model'):
            idx = data[data['Car Model'].str.lower() == preferences['car_model'].lower()].index
            if not idx.empty:
                idx = idx[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()
                similar_indices = cosine_sim.argsort()[-6:-1][::-1]  # Get top 5 similar
                return data.iloc[similar_indices].to_dict('records')

        # If no specific model but preferences exist
        query_text = ""
        if preferences.get('primary_use') == 'luxury':
            query_text = "luxury comfortable premium heated seats"
        elif preferences.get('primary_use') == 'family':
            query_text = "family safe reliable spacious kids"
        elif preferences.get('primary_use') == 'commuter':
            query_text = "commuter efficient mpg mileage economical"
        elif preferences.get('primary_use') == 'performance':
            query_text = "performance fast quick handling power"

        if query_text:
            query_vec = tfidf.transform([query_text])
            cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
            similar_indices = cosine_sim.argsort()[-5:][::-1]  # Get top 5
            return data.iloc[similar_indices].to_dict('records')

        return []
    except Exception as e:
        print(f"Error in content_based_recommendation: {str(e)}")
        return []


# Memory-Based Recommendation (kNN)
def memory_based_recommendation(car_model):
    # Use ratings for collaborative filtering
    rating_columns = ['Comfort', 'Interior', 'Performance', 'Value', 'Exterior', 'Reliability']
    X = data[rating_columns].values
    knn = NearestNeighbors(n_neighbors=4, metric='cosine')
    knn.fit(X)
    # Find index of the car model
    idx = data[data['Car Model'].str.lower() == car_model.lower()].index
    if not idx.empty:
        idx = idx[0]
        # Find nearest neighbors
        distances, indices = knn.kneighbors([X[idx]])
        # Exclude the input car
        indices = indices[0][1:]
        return data.iloc[indices][['Brand', 'Car Model', 'Rating']].to_dict('records')
    return []


# Model-Based Recommendation (SVD)
def model_based_recommendation(car_model):
    # Create user-item matrix (pivot on Reviewer for simplicity)
    user_item_matrix = data.pivot_table(index='Reviewer', columns='Car Model', values='Rating')
    user_item_matrix = user_item_matrix.fillna(0)
    # Apply SVD
    svd = TruncatedSVD(n_components=5)
    matrix = svd.fit_transform(user_item_matrix)
    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix)
    # Find index of the car model
    car_idx = user_item_matrix.columns.get_loc(car_model) if car_model in user_item_matrix.columns else -1
    if car_idx != -1:
        # Get top 3 correlated cars
        corr_scores = corr_matrix[car_idx]
        top_indices = corr_scores.argsort()[-4:-1][::-1]
        top_cars = user_item_matrix.index[top_indices]
        # Filter original data for these reviewers
        return data[data['Reviewer'].isin(top_cars)][['Brand', 'Car Model', 'Rating']].head(3).to_dict('records')
    return []


# Flask Routes
@app.route('/')
def index():
    brands = data['Brand'].unique().tolist()
    car_models = data['Car Model'].unique().tolist()
    return render_template('index.html', brands=brands, car_models=car_models)


@app.route('/recommend', methods=['POST'])
def recommend():
    preferences = {
        'brand': request.form.get('brand'),
        'car_model': request.form.get('car_model'),
        'primary_use': request.form.get('primary_use'),
        'min_reliability': request.form.get('min_reliability'),
        'min_comfort': request.form.get('min_comfort')
    }

    # Generate recommendations
    rule_based = rule_based_recommendation(preferences)
    content_based = content_based_recommendation(preferences['car_model']) if preferences['car_model'] else []
    memory_based = memory_based_recommendation(preferences['car_model']) if preferences['car_model'] else []
    model_based = model_based_recommendation(preferences['car_model']) if preferences['car_model'] else []

    # Prepare comparison data
    def get_comparison_metrics(recs):
        if not recs:
            return {'count': 0, 'diversity': 0, 'relevance': 'N/A'}

        # Extract unique models
        if isinstance(recs[0], dict):
            models = [r['Car Model'] for r in recs]
        else:
            models = [r['Car Model'] for r in recs.to_dict('records')]

        return {
            'count': len(recs),
            'diversity': len(set(models)),
            'relevance': 'High' if len(recs) > 0 else 'N/A'
        }

    comparison = {
        'rule_based': get_comparison_metrics(rule_based),
        'content_based': get_comparison_metrics(content_based),
        'memory_based': get_comparison_metrics(memory_based),
        'model_based': get_comparison_metrics(model_based)
    }

    return render_template('results.html',
                           rule_based=rule_based,
                           content_based=content_based,
                           memory_based=memory_based,
                           model_based=model_based,
                           comparison=comparison,
                           preferences=preferences)


if __name__ == '__main__':
    app.run(debug=True)