# Movie Recommendation Script

A simple item-based collaborative filtering script that reads user/movie ratings, computes cosine similarity between movies, predicts missing ratings, and prints the top recommendations for each user.

## Requirements
- Python 3.8+
- `pandas`, `numpy`

Install dependencies:
```bash
pip install -r requirements.txt  # or pip install pandas numpy
```

## Dataset
Expected CSV columns: `user_id`, `movie_id`, `rating`, `timestamp`.  
Example file included: `train_ratings.csv` (up to 6040 users and 3675 movies).

## Usage
```bash
python3 recommender.py <ratings.csv> <similarity_threshold>
```

Example:
```bash
python3 recommender.py train_ratings.csv 0.5
```

At runtime, you will be prompted for:
- Number of users to include (clamped to dataset size)
- Number of movies to include (clamped to dataset size)

Output: for each selected user, the script prints the top 25 recommended movie IDs with their predicted ratings, followed by execution time.

## Notes
- Similarity uses cosine distance on rating vectors.
- Ratings and similarities below the provided threshold are ignored in predictions.
- The double loop over movies is quadratic; large subsets can be slow. Start with smaller counts if performance is an issue.
