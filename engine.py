import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Loading valid movies...")
df = pd.read_pickle("movies_with_vectors.pkl")

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations_for_user(watched_movies):
    """
    Takes a LIST of movie titles, creates a User Vector, 
    and returns top 5 recommendations.
    """
    
    valid_vectors = []
    
    for movie in watched_movies:
        if movie in indices:
            idx = indices[movie]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            valid_vectors.append(df.iloc[idx]['vector'])
        else:
            print(f"Warning: '{movie}' not found in database. Skipping.")

    if not valid_vectors:
        return "Error: None of the selected movies were found."

    user_vector = np.mean(valid_vectors, axis=0)
    user_vector = user_vector.reshape(1, -1)
    
    all_vectors = list(df['vector'])
    sim_scores = cosine_similarity(user_vector, all_vectors)
    
    sim_scores_list = list(enumerate(sim_scores[0]))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)
    
    watched_indices = [indices[m] for m in watched_movies if m in indices]
    
    recommendations = []
    for i, score in sim_scores_list:
        if len(recommendations) >= 5:
            break
            
        is_watched = False
        if isinstance(watched_indices, list):
             for watched_idx in watched_indices:
                 current_idx = watched_idx.iloc[0] if isinstance(watched_idx, pd.Series) else watched_idx
                 if i == current_idx:
                     is_watched = True
        
        if not is_watched:
            recommendations.append(df.iloc[i]['title'])
            
    return recommendations

my_watched_history = ["Love Untangled", "Frankenstein", "Bird Box", "Alive"]

print(f"\nUser Watched: {my_watched_history}")
print("-" * 40)
recs = get_recommendations_for_user(my_watched_history)
print("We think you will like:")
for movie in recs:
    print(f"- {movie}")