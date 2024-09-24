# Movie Recommendation System

## Overview
This project is an end-to-end movie recommendation system that allows users to filter movies based on various criteria such as release date, genre, ratings, and title. It utilizes over 110,000 movie entries from the TMDB dataset, providing a rich and diverse selection for recommendations. The system has been designed to be user-friendly, allowing easy interaction through a web interface powered by Streamlit and Gradio.

## Features
- **Filtering Options**: Users can filter movies by:
  - Release Date
  - Genre
  - Ratings (Vote Average)
  - Movie Title
- **Recommendation Engine**: Provides personalized movie suggestions based on the user's input.
- **User Interface**: Built with Streamlit and Gradio for an interactive user experience.

## Dataset
The dataset used for this project is the TMDB Movies Dataset, which contains detailed information about movies, including their titles, genres, release dates, popularity, and ratings. You can access the dataset [here](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/code).

## Installation
To get started, ensure you have Python installed on your system. You can clone this repository and install the required libraries:

```bash
git clone https://github.com/roybishal362/End---to---End-Movie-recommended-System.git
cd End---to---End-Movie-recommended-System
pip install -r requirements.txt

## Usage
1. Load the dataset and prepare the model.
2. Run the Streamlit application to start the web interface.

   ```bash
   streamlit run app.py


Alternatively, you can use the Gradio interface:
```bash
!pip install gradio
python gradio_app.py

