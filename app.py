"""
Flask Web Application for Movie Recommendation System
"""
from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALSModel
import os
import json

app = Flask(__name__)

# Global variables for Spark session and model
spark = None
model = None
movies_df = None
ratings_df = None

def initialize_spark():
    """Initialize Spark session and load model"""
    global spark, model, movies_df, ratings_df
    
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("MovieRecommendationWeb") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load movies data
    print("Loading movie data...")
    movies_df = spark.read.csv("data/movies.csv", header=True, inferSchema=True) \
        .select("movieId", "title", "genres")
    
    # Load ratings data
    ratings_df = spark.read.csv("data/ratings.csv", header=True, inferSchema=True) \
        .select("userId", "movieId", "rating")
    
    # Try to load existing model, or train a new one
    model_path = "models/als_movie_model"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = ALSModel.load(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            model = train_new_model()
    else:
        print("No existing model found. Training new model...")
        model = train_new_model()
    
    return spark, model, movies_df

def train_new_model():
    """Train a new ALS model"""
    from pyspark.ml.recommendation import ALS
    
    print("Training ALS model (this may take a few minutes)...")
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
              rank=10, regParam=0.1, maxIter=10,
              coldStartStrategy="drop", nonnegative=True)
    
    trained_model = als.fit(ratings_df)
    
    # Save the model
    try:
        os.makedirs("models", exist_ok=True)
        trained_model.write().overwrite().save("models/als_movie_model")
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
    
    return trained_model

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/users')
def get_users():
    """Get list of available user IDs"""
    try:
        user_ids = ratings_df.select("userId").distinct().orderBy("userId").limit(100).collect()
        users = [row["userId"] for row in user_ids]
        return jsonify({"users": users, "total": len(users)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/movies')
def get_movies():
    """Get list of all movies"""
    try:
        limit = request.args.get('limit', 50, type=int)
        search = request.args.get('search', '', type=str)
        
        if search:
            movies = movies_df.filter(col("title").contains(search)).limit(limit).collect()
        else:
            movies = movies_df.limit(limit).collect()
        
        movie_list = [
            {
                "movieId": row["movieId"],
                "title": row["title"],
                "genres": row["genres"]
            }
            for row in movies
        ]
        return jsonify({"movies": movie_list, "count": len(movie_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/<int:user_id>')
def recommend_for_user(user_id):
    """Get recommendations for a specific user"""
    try:
        n = request.args.get('n', 10, type=int)
        
        # Create a DataFrame with the user ID
        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        
        # Get recommendations
        user_recs = model.recommendForUserSubset(user_df, n)
        
        # Check if we got any recommendations
        if user_recs.count() == 0:
            return jsonify({"error": f"No recommendations found for user {user_id}"}), 404
        
        # Explode and join with movie titles
        exploded = user_recs.select("userId", explode(col("recommendations")).alias("rec"))
        exploded = exploded.select(
            col("userId"),
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("predicted_rating")
        )
        
        # Join with movies to get titles and genres
        joined = exploded.join(movies_df, on="movieId", how="left") \
                        .select("userId", "movieId", "title", "genres", "predicted_rating")
        
        # Convert to list
        recommendations = [
            {
                "movieId": row["movieId"],
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": float(row["predicted_rating"])
            }
            for row in joined.collect()
        ]
        
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
        total_users = ratings_df.select("userId").distinct().count()
        total_movies = movies_df.count()
        total_ratings = ratings_df.count()
        
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
        movie = movies_df.filter(col("movieId") == movie_id).collect()
        
        if not movie:
            return jsonify({"error": f"Movie {movie_id} not found"}), 404
        
        movie_data = movie[0]
        
        # Get rating statistics for this movie
        movie_ratings = ratings_df.filter(col("movieId") == movie_id)
        avg_rating = movie_ratings.agg({"rating": "avg"}).collect()[0][0]
        rating_count = movie_ratings.count()
        
        return jsonify({
            "movieId": movie_data["movieId"],
            "title": movie_data["title"],
            "genres": movie_data["genres"],
            "average_rating": float(avg_rating) if avg_rating else 0,
            "rating_count": rating_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize Spark and load model before starting the server
    initialize_spark()
    
    print("\n" + "="*60)
    print("🎬 Movie Recommendation System - Web Server Starting...")
    print("="*60)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
