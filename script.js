// API Base URL
const API_BASE = '/api';

// DOM Elements
const userSelect = document.getElementById('userSelect');
const numRecsInput = document.getElementById('numRecs');
const getRecsBtn = document.getElementById('getRecsBtn');
const recommendationsPanel = document.getElementById('recommendationsPanel');
const recommendationsList = document.getElementById('recommendationsList');
const currentUserIdSpan = document.getElementById('currentUserId');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const searchMoviesInput = document.getElementById('searchMovies');
const moviesList = document.getElementById('moviesList');

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadUsers();
    loadMovies();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    getRecsBtn.addEventListener('click', getRecommendations);
    
    // Search movies with debounce
    let searchTimeout;
    searchMoviesInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            loadMovies(e.target.value);
        }, 500);
    });
}

// Load system statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading stats:', data.error);
            return;
        }
        
        document.getElementById('totalUsers').textContent = data.total_users.toLocaleString();
        document.getElementById('totalMovies').textContent = data.total_movies.toLocaleString();
        document.getElementById('totalRatings').textContent = data.total_ratings.toLocaleString();
        
        // Animate numbers
        animateNumber('totalUsers', data.total_users);
        animateNumber('totalMovies', data.total_movies);
        animateNumber('totalRatings', data.total_ratings);
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Animate number counting
function animateNumber(elementId, finalValue) {
    const element = document.getElementById(elementId);
    const duration = 1000;
    const steps = 30;
    const increment = finalValue / steps;
    let current = 0;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= finalValue) {
            element.textContent = finalValue.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, duration / steps);
}

// Load available users
async function loadUsers() {
    try {
        const response = await fetch(`${API_BASE}/users`);
        const data = await response.json();
        
        if (data.error) {
            showError('Error loading users: ' + data.error);
            return;
        }
        
        userSelect.innerHTML = '<option value="">-- Select a User --</option>';
        data.users.forEach(userId => {
            const option = document.createElement('option');
            option.value = userId;
            option.textContent = `User ${userId}`;
            userSelect.appendChild(option);
        });
    } catch (error) {
        showError('Error loading users: ' + error.message);
    }
}

// Load movies
async function loadMovies(searchTerm = '') {
    try {
        const url = searchTerm 
            ? `${API_BASE}/movies?search=${encodeURIComponent(searchTerm)}&limit=50`
            : `${API_BASE}/movies?limit=50`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading movies:', data.error);
            return;
        }
        
        displayMovies(data.movies);
    } catch (error) {
        console.error('Error loading movies:', error);
    }
}

// Display movies in browser
function displayMovies(movies) {
    if (!movies || movies.length === 0) {
        moviesList.innerHTML = '<p style="text-align: center; color: #666;">No movies found</p>';
        return;
    }
    
    moviesList.innerHTML = movies.map(movie => `
        <div class="movie-item">
            <div class="movie-item-title">${escapeHtml(movie.title)}</div>
            <div class="movie-item-genres">${escapeHtml(movie.genres || 'N/A')}</div>
        </div>
    `).join('');
}

// Get recommendations for selected user
async function getRecommendations() {
    const userId = userSelect.value;
    const numRecs = parseInt(numRecsInput.value) || 10;
    
    if (!userId) {
        showError('Please select a user first!');
        return;
    }
    
    // Hide previous results and errors
    hideError();
    recommendationsPanel.style.display = 'none';
    loadingSpinner.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/recommend/${userId}?n=${numRecs}`);
        const data = await response.json();
        
        loadingSpinner.style.display = 'none';
        
        if (data.error) {
            showError('Error: ' + data.error);
            return;
        }
        
        displayRecommendations(data);
    } catch (error) {
        loadingSpinner.style.display = 'none';
        showError('Error getting recommendations: ' + error.message);
    }
}

// Display recommendations
function displayRecommendations(data) {
    currentUserIdSpan.textContent = data.userId;
    
    if (!data.recommendations || data.recommendations.length === 0) {
        recommendationsList.innerHTML = '<p style="text-align: center; color: #666;">No recommendations available</p>';
        recommendationsPanel.style.display = 'block';
        return;
    }
    
    recommendationsList.innerHTML = data.recommendations.map((movie, index) => {
        const stars = getStarRating(movie.predicted_rating);
        return `
            <div class="movie-card" style="animation-delay: ${index * 0.1}s">
                <div>
                    <span class="movie-rank">#${index + 1}</span>
                    <span class="movie-title">${escapeHtml(movie.title)}</span>
                </div>
                <div class="movie-genres">
                    📁 ${escapeHtml(movie.genres || 'N/A')}
                </div>
                <div class="movie-rating">
                    <span class="rating-stars">${stars}</span>
                    Predicted Rating: ${movie.predicted_rating.toFixed(2)} / 5.0
                </div>
            </div>
        `;
    }).join('');
    
    recommendationsPanel.style.display = 'block';
    recommendationsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Convert rating to stars
function getStarRating(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    let stars = '⭐'.repeat(fullStars);
    if (hasHalfStar && fullStars < 5) stars += '⭐';
    return stars || '☆';
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Console welcome message
console.log('%c🎬 Movie Recommendation System', 'font-size: 20px; color: #667eea; font-weight: bold;');
console.log('%cPowered by Apache Spark & Machine Learning', 'font-size: 12px; color: #764ba2;');
