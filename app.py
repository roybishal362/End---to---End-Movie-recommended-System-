import streamlit as st
import pandas as pd
import pickle
import os
import zipfile
from PIL import Image
import logging
import time
from sentence_transformers import SentenceTransformer, util

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv('D:\My projects\Movie recommendation system\Data\TMDB_movie_dataset_v11.csv')
        movies_df['title'] = movies_df['title'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
        movies_df['release_year'] = movies_df['release_date'].dt.year.fillna(0).astype(int)
        movies_df['popularity'] = movies_df['popularity'].fillna(0)
        movies_df['vote_average'] = movies_df['vote_average'].fillna(0)
        return movies_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error("Failed to load movie data.")
        return pd.DataFrame()

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

# Load model and dataset
model = load_model()
movies_df = load_data()

# Function to recommend movies based on the title (using sentence transformers)
def recommend_by_title(movie_title, df, n_recommendations=10):
    query_embedding = model.encode(movie_title, convert_to_tensor=True)
    movie_embeddings = model.encode(df['title'].tolist(), convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, movie_embeddings).squeeze()
    top_results = cosine_similarities.argsort(descending=True)[:n_recommendations + 1].cpu().numpy()
    recommendations = [df['title'].iloc[idx] for idx in top_results if df['title'].iloc[idx].lower() != movie_title.lower()][:n_recommendations]
    return recommendations

# Function to extract images from a ZIP file
def extract_images(zip_path, extract_to='posters'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to display a movie image slideshow
def display_slideshow(image_folder, size=(300, 450)):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        st.warning("No images found in the specified folder.")
        return

    current_index = 0
    image_count = len(image_files)
    slideshow_placeholder = st.empty()

    while True:
        image_path = os.path.join(image_folder, image_files[current_index])
        try:
            image = Image.open(image_path).resize(size)
            with slideshow_placeholder.container():
                st.image(image, use_column_width='auto')
        except Exception as e:
            logging.error(f"Error loading image {image_files[current_index]}: {e}")

        current_index = (current_index + 1) % image_count  # Loop back to the first image
        time.sleep(3)  # Change image every 3 seconds

# Function to filter movies
def filter_movies(movies_df, start_year, end_year, min_popularity, min_vote):
    filtered_df = movies_df[
        (movies_df['release_year'] >= start_year) &
        (movies_df['release_year'] <= end_year) &
        (movies_df['popularity'] >= min_popularity) &
        (movies_df['vote_average'] >= min_vote)
    ]
    return filtered_df

# Streamlit app UI
def movie_recommendation_app():
    st.set_page_config(page_title="Netflix-Like Movie Recommendation System", layout="wide")
    st.title("🎬 Netflix-Like Movie Recommendation System")

    # Extract movie posters from ZIP file
    zip_path = "D:\My projects\Movie recommendation system\posters.zip"  # Adjust this path
    extract_images(zip_path)

    # Display slideshow of movie posters
    st.header("🎥 Featured Movie Posters Slideshow")
    poster_folder = "posters"  # The folder where posters are extracted
    display_slideshow(poster_folder, size=(300, 450))

    # Sidebar filters
    st.sidebar.header("Filters")
    title_input = st.sidebar.text_input("Search by Movie Title", placeholder="Enter movie title here...")
    start_year = st.sidebar.slider("Start Year", min_value=1900, max_value=2024, value=2000)
    end_year = st.sidebar.slider("End Year", min_value=1900, max_value=2024, value=2020)
    min_popularity = st.sidebar.slider("Minimum Popularity", min_value=0, max_value=100, value=50)
    min_vote = st.sidebar.slider("Minimum Vote Average", min_value=0.0, max_value=10.0, value=7.0)

    # Filter movies
    filtered_df = filter_movies(movies_df, start_year, end_year, min_popularity, min_vote)

    if title_input:
        recommendations = recommend_by_title(title_input, filtered_df)
        if recommendations:
            st.subheader(f"Top recommendations based on '{title_input}':")
            for movie_title in recommendations:
                movie_info = filtered_df[filtered_df['title'] == movie_title].iloc[0]
                st.markdown(f"""
                <div style='display: inline-block; margin: 10px; text-align: center;'>
                    <img src='posters/{movie_info["poster_path"]}' style='width: 150px; height: 225px;'>
                    <br><strong>{movie_info['title']}</strong>
                    <br>Release Year: {movie_info['release_year']}
                    <br>Rating: {movie_info['vote_average']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No recommendations found.")
    else:
        st.write("Use the filters or search by title to get movie recommendations.")

# Run the app
if __name__ == "__main__":
    movie_recommendation_app()
