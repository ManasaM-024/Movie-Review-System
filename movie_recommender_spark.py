# movie_recommender_spark.py
"""
Movie Recommendation System using PySpark ALS.

Requirements:
  - PySpark installed (pyspark)
  - MovieLens data files:
      data/ratings.csv  (columns: userId,movieId,rating,timestamp)
      data/movies.csv   (columns: movieId,title,genres)

Run:
  spark-submit movie_recommender_spark.py
  OR
  python movie_recommender_spark.py   # when running locally with pyspark available
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import os

def create_spark_session(app_name="MovieRecommendationALS"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark, ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True) \
        .select("userId", "movieId", "rating")
    movies = spark.read.csv(movies_path, header=True, inferSchema=True) \
        .select("movieId", "title")
    return ratings, movies

def train_als_model(ratings_df, rank=10, regParam=0.1, maxIter=10):
    """
    Train basic ALS model (no CV).
    """
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
              rank=rank, regParam=regParam, maxIter=maxIter,
              coldStartStrategy="drop", nonnegative=True)
    model = als.fit(ratings_df)
    return model

def train_als_with_tuning(train_df):
    """
    Train ALS using TrainValidationSplit to tune hyperparameters.
    Returns best model and the TVS object (contains metrics).
    """
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", nonnegative=True)

    paramGrid = ParamGridBuilder() \
        .addGrid(als.rank, [8, 12, 16]) \
        .addGrid(als.regParam, [0.05, 0.1, 0.2]) \
        .addGrid(als.maxIter, [8, 12]) \
        .build()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    tvs = TrainValidationSplit(estimator=als,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               trainRatio=0.8,
                               parallelism=2)  # adjust parallelism as per your environment

    tvs_model = tvs.fit(train_df)
    best_model = tvs_model.bestModel
    return best_model, tvs_model

def evaluate_model(model, test_df):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    predictions = model.transform(test_df)
    rmse = evaluator.evaluate(predictions)
    return rmse, predictions

def recommend_for_all_users(model, movies_df, n=5):
    """
    Get top-n recommendations for all users, with movie titles expanded.
    Returns a DataFrame: userId | recommendations (array of structs)
    We'll explode later to join movie titles.
    """
    user_recs = model.recommendForAllUsers(n)  # columns: userId, recommendations (array)
    # explode recommendations into rows
    exploded = user_recs.select(col("userId"), explode(col("recommendations")).alias("rec"))
    exploded = exploded.select(
        col("userId"),
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predicted_rating")
    )
    # join with movies to get titles
    joined = exploded.join(movies_df, on="movieId", how="left") \
                     .select("userId", "movieId", "title", "predicted_rating")
    return joined

def recommend_for_user(model, movies_df, user_id, n=10):
    """
    Returns top-n recommendations (with titles) for a single user.
    """
    user_df = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["userId"]), n)
    # user_df has column recommendations: array<struct(movieId, rating)>
    exploded = user_df.select("userId", explode(col("recommendations")).alias("rec"))
    exploded = exploded.select(
        col("userId"),
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predicted_rating")
    )
    joined = exploded.join(movies_df, on="movieId", how="left") \
                     .select("userId", "movieId", "title", "predicted_rating")
    return joined

if __name__ == "__main__":
    spark = create_spark_session()

    # Paths (change if needed)
    RATINGS_PATH = "data/ratings.csv"
    MOVIES_PATH = "data/movies.csv"
    MODEL_DIR = "models/als_movie_model"

    # 1) Load data
    if not (os.path.exists(RATINGS_PATH) and os.path.exists(MOVIES_PATH)):
        print("ERROR: data/ratings.csv and data/movies.csv must exist. Download MovieLens dataset and place files in data/")
        spark.stop()
        raise SystemExit(1)

    ratings, movies = load_data(spark, RATINGS_PATH, MOVIES_PATH)
    print("Total ratings:", ratings.count())
    print("Total movies:", movies.count())

    # 2) Train-test split
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)
    print("Train count:", train.count(), "Test count:", test.count())

    # 3) Train with hyperparameter tuning (TrainValidationSplit)
    print("Training ALS model with hyperparameter tuning (this may take a while)...")
    best_model, tvs_model = train_als_with_tuning(train)
    print("Best model params:")
    print("  rank =", best_model._java_obj.parent().getRank())
    print("  regParam =", best_model._java_obj.parent().getRegParam())
    print("  maxIter =", best_model._java_obj.parent().getMaxIter())

    # 4) Evaluate on test set
    rmse, predictions = evaluate_model(best_model, test)
    print(f"Test RMSE = {rmse:.4f}")

    # 5) Save the best model
    # remove existing model dir if exists (optional)
    try:
        best_model.write().overwrite().save(MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")
    except Exception as e:
        print("Warning: could not save model:", e)

    # 6) Generate top-5 recommendations for all users (and show sample)
    print("Generating top-5 recommendations for a sample of users (with titles)...")
    user_recommendations = recommend_for_all_users(best_model, movies, n=5)
    user_recommendations.show(20, truncate=50)

    # 7) Recommend for a specific user (pick one from dataset)
    some_user_id_row = ratings.select("userId").distinct().limit(1).collect()
    some_user_id = some_user_id_row[0]["userId"]
    print(f"Sample user id for demonstration: {some_user_id}")

    user_recs = best_model.recommendForUserSubset(spark.createDataFrame([(some_user_id,)], ["userId"]), 10)
    # Display with titles for readability
    exploded = user_recs.select("userId", explode(col("recommendations")).alias("rec"))
    exploded = exploded.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("predicted_rating"))
    exploded = exploded.join(movies, on="movieId", how="left").select("userId", "movieId", "title", "predicted_rating")
    print(f"Top recommendations for user {some_user_id}:")
    exploded.show(truncate=50)

    # 8) Stop Spark
    spark.stop()