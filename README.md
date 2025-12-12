# 🎬 Movie Recommendation System - Web Application

A beautiful web-based movie recommendation system powered by Apache Spark's ALS (Alternating Least Squares) machine learning algorithm.

## Features

- **Real-time Recommendations**: Get personalized movie recommendations for any user
- **Interactive Dashboard**: View system statistics (total users, movies, ratings)
- **Movie Browser**: Search and explore the movie database
- **Beautiful UI**: Modern, responsive design with smooth animations
- **REST API**: Full-featured API for recommendations and movie data

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation & Setup

1. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Verify Data Files**
   Make sure you have the following data files in the `data/` directory:
   - `data/ratings.csv` (userId, movieId, rating, timestamp)
   - `data/movies.csv` (movieId, title, genres)

## Running the Application

### Step 1: Start the Web Server

```powershell
python app.py
```

The first time you run the application:
- Spark will initialize (takes 10-30 seconds)
- If no model exists, it will train a new ALS model (takes 2-5 minutes depending on dataset size)
- The model will be saved to `models/als_movie_model/` for future use

### Step 2: Open Your Browser

Once you see the message:
```
🎬 Movie Recommendation System - Web Server Starting...
Open your browser and go to: http://localhost:5000
```

Navigate to: **http://localhost:5000**

### Step 3: Use the Application

1. **View Statistics**: See total users, movies, and ratings in the dashboard
2. **Get Recommendations**:
   - Select a user ID from the dropdown
   - Choose number of recommendations (1-50)
   - Click "Get Recommendations"
3. **Browse Movies**: Use the search box to find specific movies

## API Endpoints

The application provides the following REST API endpoints:

- `GET /api/stats` - Get system statistics
- `GET /api/users` - Get list of user IDs
- `GET /api/movies?search=<term>&limit=<n>` - Search movies
- `GET /api/recommend/<user_id>?n=<count>` - Get recommendations for a user
- `GET /api/movie/<movie_id>` - Get movie details

## Project Structure

```
movierecomendation/
├── app.py                          # Flask web application
├── movie_recommender_spark.py      # Original Spark training script
├── requirements.txt                # Python dependencies
├── data/
│   ├── movies.csv                  # Movie metadata
│   └── ratings.csv                 # User ratings
├── models/
│   └── als_movie_model/           # Trained ALS model (auto-generated)
├── templates/
│   └── index.html                 # Web UI template
└── static/
    ├── style.css                  # Stylesheets
    └── script.js                  # Frontend JavaScript
```

## How It Works

1. **Data Loading**: Loads movie ratings and metadata from CSV files
2. **Model Training**: Uses Spark's ALS algorithm to learn user preferences
3. **Predictions**: Generates personalized recommendations based on collaborative filtering
4. **Web Interface**: Provides an easy-to-use interface for exploring recommendations

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
```

### Model Training Takes Too Long
For faster initial setup, you can run the original training script separately:
```powershell
python movie_recommender_spark.py
```

### Java Not Found
PySpark requires Java. Install Java JDK 8 or 11 if you get Java-related errors.

## Stopping the Server

Press `CTRL+C` in the terminal to stop the web server.

## Technologies Used

- **Backend**: Python, Flask, PySpark
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Machine Learning**: Apache Spark MLlib (ALS algorithm)
- **Data**: MovieLens dataset

---

Built with ❤️ using Apache Spark & Machine Learning
