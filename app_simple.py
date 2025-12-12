"""
Flask Web Application for Movie Recommendation System (Simplified Version)
Works without PySpark - Uses pandas and scikit-learn
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Global variables
movies_df = None
ratings_df = None
user_item_matrix = None
item_similarity = None

def initialize_data():
    """Load and prepare data"""
    global movies_df, ratings_df, user_item_matrix, item_similarity
    
    print("Loading movie data...")
    
    # Load movies (with proper escaping for commas in titles)
    movies_df = pd.read_csv("data/movies.csv", encoding='utf-8', on_bad_lines='skip')
    print(f"Loaded {len(movies_df)} movies")
    
    # Load ratings
    ratings_df = pd.read_csv("data/ratings.csv", encoding='utf-8')
    print(f"Loaded {len(ratings_df)} ratings")
    
    # Create user-item matrix
    print("Creating recommendation model...")
    user_item_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    
    # Calculate item-item similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    
    print("Model ready!")
    return movies_df, ratings_df

def get_recommendations_for_user(user_id, n=10):
    """Get recommendations for a specific user"""
    if user_id not in user_item_matrix.index:
        return []
    
    # Get user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Find movies user hasn't rated
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Calculate predicted ratings based on similarity
    predictions = {}
    for movie_id in unrated_movies:
        if movie_id not in user_item_matrix.columns:
            continue
            
        # Get similar movies that user has rated
        similar_movies = user_item_matrix.columns[user_ratings > 0]
        
        if len(similar_movies) == 0:
            continue
        
        # Calculate weighted average
        similarities = item_similarity[
            user_item_matrix.columns.get_loc(movie_id),
            [user_item_matrix.columns.get_loc(m) for m in similar_movies if m in user_item_matrix.columns]
        ]
        
        ratings = [user_ratings[m] for m in similar_movies if m in user_item_matrix.columns]
        
        if len(similarities) > 0 and sum(abs(similarities)) > 0:
            predicted_rating = np.dot(similarities, ratings) / sum(abs(similarities))
            predictions[movie_id] = predicted_rating
    
    # Sort by predicted rating
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations with movie details
    recommendations = []
    for movie_id, predicted_rating in sorted_predictions[:n]:
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_info.iloc[0]['title'],
                'genres': movie_info.iloc[0]['genres'],
                'predicted_rating': float(predicted_rating)
            })
    
    return recommendations

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/users')
def get_users():
    """Get list of available user IDs"""
    try:
        user_ids = sorted(ratings_df['userId'].unique())[:100]
        return jsonify({"users": [int(u) for u in user_ids], "total": len(user_ids)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/movies')
def get_movies():
    """Get list of all movies"""
    try:
        limit = request.args.get('limit', 50, type=int)
        search = request.args.get('search', '', type=str)
        
        if search:
            filtered = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
            movies = filtered.head(limit)
        else:
            movies = movies_df.head(limit)
        
        movie_list = [
            {
                "movieId": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres']
            }
            for _, row in movies.iterrows()
        ]
        return jsonify({"movies": movie_list, "count": len(movie_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/<int:user_id>')
def recommend_for_user(user_id):
    """Get recommendations for a specific user"""
    try:
        n = request.args.get('n', 10, type=int)
        
        recommendations = get_recommendations_for_user(user_id, n)
        
        if not recommendations:
            return jsonify({"error": f"No recommendations found for user {user_id}"}), 404
        
        return jsonify({
            "userId": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        total_users = int(ratings_df['userId'].nunique())
        total_movies = int(len(movies_df))
        total_ratings = int(len(ratings_df))
        
        return jsonify({
            "total_users": total_users,
            "total_movies": total_movies,
            "total_ratings": total_ratings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/movie/<int:movie_id>')
def get_movie_details(movie_id):
    """Get details for a specific movie"""
    try:
        movie = movies_df[movies_df['movieId'] == movie_id]
        
        if movie.empty:
            return jsonify({"error": f"Movie {movie_id} not found"}), 404
        
        movie_data = movie.iloc[0]
        
        # Get rating statistics for this movie
        movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
        avg_rating = float(movie_ratings['rating'].mean()) if len(movie_ratings) > 0 else 0
        rating_count = int(len(movie_ratings))
        
        return jsonify({
            "movieId": int(movie_data['movieId']),
            "title": movie_data['title'],
            "genres": movie_data['genres'],
            "average_rating": avg_rating,
            "rating_count": rating_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize data before starting the server
    initialize_data()
    
    print("\n" + "="*60)
    print("🎬 Movie Recommendation System - Web Server Starting...")
    print("="*60)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
