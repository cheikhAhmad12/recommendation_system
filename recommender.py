#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import time

# Fonction de similarité cosinus
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# Prédiction de la note
def predict_rating(user_id, movie_id, user_movie_matrix, movie_similarity, treshold):
    user_ratings = user_movie_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index
    similarities = movie_similarity.loc[movie_id, rated_movies]
    similarities = similarities[similarities > treshold]
    weighted_ratings = 0
    similarity_sum = 0
    for rated_movie in similarities.index:
        similarity = similarities[rated_movie]
        rating = user_ratings[rated_movie]
        weighted_ratings += similarity * rating
        similarity_sum += similarity
    if similarity_sum == 0:
        return 0
    return weighted_ratings / similarity_sum

# Prédiction pour tous les films non évalués
def predict_missing_ratings(user_movie_matrix, movie_similarity, treshold):
    predictions = {}
    for user_id in user_movie_matrix.index:
        predictions[user_id] = {}
        for movie_id in user_movie_matrix.columns:
            if user_movie_matrix.loc[user_id, movie_id] == 0:
                predicted_rating = predict_rating(user_id, movie_id, user_movie_matrix, movie_similarity, treshold)
                predictions[user_id][movie_id] = predicted_rating
    return predictions

# Génération des recommandations
def gen_recommendations(predicted_ratings, top_n=5):
    recommendations = {}
    for user_id, movie_ratings in predicted_ratings.items():
        sorted_ratings = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = sorted_ratings[:top_n]
        recommendations[user_id] = top_recommendations
    return recommendations

def main(file_path, treshold):
    start_time = time.time()
    ratings_df = pd.read_csv(file_path)
    num_users= int(input("Number of users : "))
    num_movies= int(input("Number of movies : "))
    if num_users < 0 :
        num_users=0
    if num_users > 6040 :
        num_users= 6040
    if num_movies <0 : 
        num_movies =0
    if num_movies >3675:
        num_movies=3675
    user_movie_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_movie_matrix = user_movie_matrix.iloc[:num_users, :num_movies]

    movie_ids = user_movie_matrix.columns
    movie_similarity = pd.DataFrame(index=movie_ids, columns=movie_ids)

    for i in movie_ids:
        for j in movie_ids:
            if i != j:
                movie_similarity.loc[i, j] = cosine_similarity(user_movie_matrix[i], user_movie_matrix[j])
            else:
                movie_similarity.loc[i, j] = 1

    predicted_ratings = predict_missing_ratings(user_movie_matrix, movie_similarity, treshold)
    recommendations = gen_recommendations(predicted_ratings, top_n=25)
    for user_id, user_recommendations in recommendations.items():
        for movie_id, predicted_rating in user_recommendations:
            print(f"user {user_id} movie {movie_id} note {predicted_rating:.2f}")
        print("\n") 
    end_time = time.time()
    execution_time = end_time - start_time
    print("Recommandations générées en", execution_time, "secondes.")
    print("Exécution terminée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender system script.")
    parser.add_argument("file_path", type=str, help="Chemin vers le fichier de notation des utilisateurs")
    parser.add_argument("treshold", type=float, help="Seuil de similarité pour les recommandations")
    args = parser.parse_args()

    main(args.file_path, args.treshold)