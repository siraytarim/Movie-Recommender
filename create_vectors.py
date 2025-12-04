import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("movies_dataset.csv")
df['overview'] = df['overview'].fillna('')

print("Downloading/Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Converting text to numbers (Vectors)... this might take a moment.")
vectors = model.encode(df['overview'].tolist(), show_progress_bar=True)

df['vector'] = list(vectors)

df.to_pickle("movies_with_vectors.pkl")

print(f"SUCCESS! Created 'movies_with_vectors.pkl' with {len(df)} rows.")
print(f"Each movie is now represented by {len(df['vector'][0])} numbers.")