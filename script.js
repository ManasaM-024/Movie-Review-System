// Spark Movie Reviews - JavaScript Functionality

// Sample initial data
let movies = [
  {
    id: 1,
    title: "Inception",
    rating: 5,
    comments: "Mind-bending masterpiece with incredible visuals and storytelling.",
    date: new Date('2023-05-15')
  },
  {
    id: 2,
    title: "The Shawshank Redemption",
    rating: 5,
    comments: "Timeless classic with powerful performances and an uplifting story.",
    date: new Date('2023-06-20')
  },
  {
    id: 3,
    title: "Interstellar",
    rating: 4,
    comments: "Visually stunning space epic with emotional depth. Some pacing issues.",
    date: new Date('2023-07-10')
  }
];

let currentId = 4; // For generating new IDs
let isEditing = false;
let editId = null;

// DOM Elements
const movieForm = document.getElementById('movieForm');
const moviesList = document.getElementById('moviesList');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const filterRating = document.getElementById('filterRating');
const sortBy = document.getElementById('sortBy');
const formTitle = document.getElementById('formTitle');
const saveBtn = document.getElementById('saveBtn');
const cancelBtn = document.getElementById('cancelBtn');

// Rating stars
const stars = document.querySelectorAll('.star');
const ratingInput = document.getElementById('rating');

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
  renderMovies();
  setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
  // Form submission
  movieForm.addEventListener('submit', handleFormSubmit);
  
  // Cancel button
  cancelBtn.addEventListener('click', resetForm);
  
  // Search functionality
  searchBtn.addEventListener('click', filterAndSortMovies);
  searchInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') {
      filterAndSortMovies();
    }
  });
  
  // Filter and sort
  filterRating.addEventListener('change', filterAndSortMovies);
  sortBy.addEventListener('change', filterAndSortMovies);
  
  // Rating stars
  stars.forEach(star => {
    star.addEventListener('click', () => {
      const rating = star.getAttribute('data-rating');
      setRating(rating);
    });
    
    star.addEventListener('mouseover', () => {
      const rating = star.getAttribute('data-rating');
      highlightStars(rating);
    });
    
    star.addEventListener('mouseout', () => {
      const currentRating = ratingInput.value;
      highlightStars(currentRating);
    });
  });
}

// Handle form submission
function handleFormSubmit(e) {
  e.preventDefault();
  
  const title = document.getElementById('title').value;
  const rating = ratingInput.value;
  const comments = document.getElementById('comments').value;
  
  if (!title || !rating) {
    alert('Please fill in all required fields');
    return;
  }
  
  if (isEditing) {
    // Update existing movie
    const index = movies.findIndex(movie => movie.id === editId);
    if (index !== -1) {
      movies[index].title = title;
      movies[index].rating = parseInt(rating);
      movies[index].comments = comments;
      movies[index].date = new Date(); // Update date
    }
  } else {
    // Add new movie
    const newMovie = {
      id: currentId++,
      title,
      rating: parseInt(rating),
      comments,
      date: new Date()
    };
    movies.push(newMovie);
  }
  
  renderMovies();
  resetForm();
}

// Set rating value and highlight stars
function setRating(rating) {
  ratingInput.value = rating;
  highlightStars(rating);
}

// Highlight stars up to the given rating
function highlightStars(rating) {
  stars.forEach(star => {
    const starRating = star.getAttribute('data-rating');
    if (starRating <= rating) {
      star.classList.add('active');
    } else {
      star.classList.remove('active');
    }
  });
}

// Reset form to default state
function resetForm() {
  movieForm.reset();
  ratingInput.value = '';
  highlightStars(0);
  formTitle.textContent = 'Add New Movie Review';
  saveBtn.textContent = 'Save Review';
  isEditing = false;
  editId = null;
}

// Edit a movie
function editMovie(id) {
  const movie = movies.find(movie => movie.id === id);
  if (!movie) return;
  
  // Fill form with movie data
  document.getElementById('title').value = movie.title;
  setRating(movie.rating);
  document.getElementById('comments').value = movie.comments;
  document.getElementById('movieId').value = movie.id;
  
  // Update form state
  isEditing = true;
  editId = id;
  formTitle.textContent = 'Edit Movie Review';
  saveBtn.textContent = 'Update Review';
  
  // Scroll to form
  document.querySelector('.movie-form-container').scrollIntoView({ behavior: 'smooth' });
}

// Delete a movie
function deleteMovie(id) {
  if (confirm('Are you sure you want to delete this movie review?')) {
    movies = movies.filter(movie => movie.id !== id);
    renderMovies();
  }
}

// Filter and sort movies
function filterAndSortMovies() {
  renderMovies();
}

// Render movies based on filters and sorting
function renderMovies() {
  // Get filter values
  const searchTerm = searchInput.value.toLowerCase();
  const ratingFilter = filterRating.value;
  const sortValue = sortBy.value;
  
  // Filter movies
  let filteredMovies = movies.filter(movie => {
    const matchesSearch = movie.title.toLowerCase().includes(searchTerm);
    const matchesRating = ratingFilter ? movie.rating == ratingFilter : true;
    return matchesSearch && matchesRating;
  });
  
  // Sort movies
  switch (sortValue) {
    case 'newest':
      filteredMovies.sort((a, b) => b.date - a.date);
      break;
    case 'oldest':
      filteredMovies.sort((a, b) => a.date - b.date);
      break;
    case 'ratingHigh':
      filteredMovies.sort((a, b) => b.rating - a.rating);
      break;
    case 'ratingLow':
      filteredMovies.sort((a, b) => a.rating - b.rating);
      break;
  }
  
  // Render movies
  if (filteredMovies.length === 0) {
    moviesList.innerHTML = `
      <div class="empty-state">
        <p>No movies found matching your criteria. Try adjusting your search or filters.</p>
      </div>
    `;
    return;
  }
  
  moviesList.innerHTML = filteredMovies.map(movie => `
    <div class="movie-card">
      <h3 class="movie-title">${movie.title}</h3>
      <div class="movie-rating">
        ${renderStars(movie.rating)}
      </div>
      <p class="movie-comments">${movie.comments}</p>
      <div class="movie-actions">
        <button class="btn-edit" onclick="editMovie(${movie.id})">Edit</button>
        <button class="btn-delete" onclick="deleteMovie(${movie.id})">Delete</button>
      </div>
    </div>
  `).join('');
}

// Render star rating
function renderStars(rating) {
  let starsHtml = '';
  for (let i = 1; i <= 5; i++) {
    if (i <= rating) {
      starsHtml += '<span class="star-filled">★</span>';
    } else {
      starsHtml += '<span class="star-empty">★</span>';
    }
  }
  return starsHtml;
}