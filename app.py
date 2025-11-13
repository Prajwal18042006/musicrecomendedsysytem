from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎵 Initialize app
app = FastAPI(title="🎵 Music Recommendation API")

# ✅ 1. Load dataset
df = pd.read_csv("tcc_ceds_music.csv")

# ✅ 2. Create combined features if not present
if 'combined_features' not in df.columns:
    df['combined_features'] = (
        df['lyrics'].fillna('') + " " +
        df['genre'].fillna('') + " " +
        df['artist_name'].fillna('')
    )

# ✅ 3. Create TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# ✅ 4. Define request schema
class SongRequest(BaseModel):
    song_name: str

# ✅ 5. Recommendation function
def recommend(song_title):
    song_title = song_title.lower()
    matches = df[df['track_name'].str.lower() == song_title]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-11:-1][::-1]
    return df[['track_name', 'artist_name', 'genre']].iloc[top_indices].to_dict(orient='records')

# ✅ 6. Home route (so you see something when visiting Render URL)
@app.get("/")
def home():
    return {"message": "🎵 Music Recommendation API is live on Render!"}

# ✅ 7. Recommendation route
@app.post("/recommend")
def get_recommendations(request: SongRequest):
    results = recommend(request.song_name)
    if not results:
        return {"error": f"Song '{request.song_name}' not found in dataset."}
    return {"recommendations": results}
