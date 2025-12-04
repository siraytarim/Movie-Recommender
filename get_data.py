import requests
import pandas as pd
import time

# 1. SETUP
headers = {
    "accept": "application/json",
    "Authorization": "Bearer (API TOKEN)" 
}

movies_list = []

print("Starting to fetch movies... this might take a minute.")

for page in range(1,500): 
    url = f"https://api.themoviedb.org/3/movie/popular?language=en-US&page={page}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        for movie in data['results']:
            movies_list.append({
                'id': movie['id'],
                'title': movie['title'],
                'overview': movie['overview'], 
                'genre_ids': movie['genre_ids'],
                'popularity': movie['popularity'],
                'vote_average': movie['vote_average'],
                'release_date': movie.get('release_date', 'N/A'),
                'poster_path': "https://image.tmdb.org/t/p/w500" + str(movie['poster_path'])
            })
    else:
        print(f"Error on page {page}")
    
    time.sleep(0.2) 

df = pd.DataFrame(movies_list)
df.to_csv("movies_dataset.csv", index=False)

print(f"Success! You have downloaded {len(df)} movies to 'movies_dataset.csv'")
print(df.head())