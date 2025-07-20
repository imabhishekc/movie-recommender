import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv

st.title('Movie Recommendation System')

load_dotenv()

TMDB_BEARER = os.getenv("TMDB_BEARER")

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_BEARER}"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


# Function to download files from Hugging Face
def download_file_from_hf(filename):
    url = f"https://huggingface.co/datasets/imabhishekc/movie-recommender-files/resolve/main/{filename}"
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print(f"{filename} downloaded.")

# Download large files if not already present
download_file_from_hf("similarity.pkl")

# Load them
movies_list = pickle.load(open('movies_dictionary.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies = pd.DataFrame(movies_list)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters


selected_movies = st.selectbox(
'What movie you would like to watch?',
movies['title'].values
)

if st.button('Recommend'):
    movie_name, posters = recommend(selected_movies)

    num_movies = len(movie_name)
    cols_per_row = 4

    for row_start in range(0, num_movies, cols_per_row):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            index = row_start + i
            if index < num_movies:
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align: center; font-size: 16px; font-weight: 600; margin-bottom: 10px;'>{movie_name[index]}</div>",
                        unsafe_allow_html=True
                    )
                    st.image(posters[index], use_container_width=True)

