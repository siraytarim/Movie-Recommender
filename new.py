import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data 
def load_data():
    df = pd.read_pickle("movies_with_vectors.pkl")
    
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    return df

try:
    df = load_data()
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
except FileNotFoundError:
    st.error("Error: 'movies_with_vectors.pkl' not found. Please run the vector script first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    year_range = st.slider(
        "Release Year Range",
        min_value=1950,
        max_value=2025,
        value=(2000, 2025) #default
    )
    st.write(f"Showing movies from **{year_range[0]}** to **{year_range[1]}**")


def get_recommendations(watched_movies, min_year, max_year):
    if not watched_movies:
        return []

    valid_vectors = []
    for movie in watched_movies:
        if movie in indices:
            idx = indices[movie]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            valid_vectors.append(df.iloc[idx]['vector'])
    
    if not valid_vectors:
        return []

    user_vector = np.mean(valid_vectors, axis=0).reshape(1, -1)
    
    sim_scores = cosine_similarity(user_vector, list(df['vector']))
    sim_scores_list = list(enumerate(sim_scores[0]))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)
    
    

    title = df.iloc[i]['title']
    print(f"Checking: {title} | Year: {movie_year} | Range: {min_year}-{max_year}")


    recommendations = []
    watched_indices = [indices[m] for m in watched_movies if m in indices]
    
    for i, score in sim_scores_list:
        if len(recommendations) >= 5:
            break

        is_watched = False
        for w_idx in watched_indices:
            val = w_idx.iloc[0] if isinstance(w_idx, pd.Series) else w_idx
            if i == val:
                is_watched = True
                    
        movie_year = df.iloc[i]['year']

        if pd.isna(movie_year):
            year_valid = False
        else:
            year_valid = (min_year <= movie_year <= max_year)
            
        if not is_watched:
            recommendations.append(df.iloc[i])
            
    return recommendations

st.title("🎬 AI Movie Recommender")
st.write("Tell us what you like, and we'll predict what you'll watch next.")

all_titles = df['title'].tolist()
selected_movies = st.multiselect(
    "Select movies you have watched:",
    options=all_titles,
    placeholder="Type to search..."
)

if st.button("Get Recommendations", type="primary"):
    if not selected_movies:
        st.warning("Please select at least one movie first!")
    else:
        with st.spinner('Calculating your taste profile...'):
           recommendations = get_recommendations(selected_movies, year_range[0], year_range[1])
        
        if recommendations:
            st.success("We think you'll love these:")
            cols = st.columns(5)
            
            for i, movie in enumerate(recommendations):
                with cols[i]:
                    poster = movie.get('poster_path')
                    if poster:
                        st.image(poster)
                    else:
                        st.image("https://via.placeholder.com/500x750?text=No+Image")
                    
                    st.markdown(f"**{movie['title']}**")
                    
                    year_display = int(movie['year']) if not pd.isna(movie['year']) else "N/A"
                    st.caption(f"📅 {year_display}")
                    
                    rating = movie.get('vote_average', 'N/A')
                    st.caption(f"⭐ {rating}/10")
                    
                    with st.expander("See Plot"):
                        st.write(movie.get('overview', 'No plot available.'))
        else:
            st.error("Could not find recommendations based on your selection.")

with st.sidebar:
    st.header("About")
    st.write("This engine uses **Vector Embeddings** to understand the semantic meaning of movie plots.")
    st.write(f"Database Size: {len(df)} movies")