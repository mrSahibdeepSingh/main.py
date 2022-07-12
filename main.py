import numpy as np
import pandas as pd


movies = pd.read_csv('tmdb_5000_movies.csv')
mcredits = pd.read_csv('tmdb_5000_credits.csv')

# print(mcredits.head(1)['cast'].values)
# print(movies['original_language'].value_counts())

# Step 1 Merge both dataframes
movies = movies.merge(mcredits, on='title')

# Step 2 select important column
# genres, id, keywords, title, overview, cast, crew

movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Step 3 create new dataframe
#id, title, tags     [tags create after merge 'overview', 'genres', 'keywords', 'cast', 'crew' these columns]

# check for missing data
movies.dropna(inplace=True)   # dropna method removes rows with null values
# print(movies.duplicated().sum())

# Perform Preprocessing
# print(movies.iloc[0].genres)
# my_obj =  '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action', 'Adventure', 'Fantasy']


import ast  # Use ast module to convert string of list to list


def convert(obj):
    my_genres = []
    for i in ast.literal_eval(obj):
        my_genres.append(i['name'])
    return my_genres


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

# convert overview column into list from string  by using split method
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# remove spaces b/w any names for genres to avoid naming confusion
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# create a new column with name tag
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Now remove all unwanted columns and create a new dataframe

new_df = movies[['id', 'title', 'tags']]

# convert tags list into string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert text into lower case
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Vectorization by using Bag of Words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()


# Stemming to convert words into their root form
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    my_list = []

    for word in text.split():
        my_list.append(ps.stem(word))

    return " ".join(my_list)


new_df['tags'] = new_df['tags'].apply(stem)

# Find distance b/w two vectors or cosine angle
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# print(recommend('Batman Begins')

# Create a Webpage and upload this code

import streamlit as st

st.title('Movie Recommend System')

option = st.selectbox(
     'Which movie would you like?',
    new_df['title'].values)

import requests


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{'
                            '}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()
    return 'https://image.tmdb.org/t/p/w500' + data['poster_path']


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies = []
    recommend_movies_poster = []
    for i in movies_list:
        movie_id = new_df.iloc[i[0]].id
        recommend_movies.append(new_df.iloc[i[0]].title)
        # Fetch movie poster from API
        recommend_movies_poster.append(fetch_poster(movie_id))

    return recommend_movies, recommend_movies_poster

from PIL import Image

if st.button('Recommended'):
    names, posters = recommend(option)

    # recommendations = recommend(option)
    for movie in names:
        st.write(movie)
    # st.write(posters[0])
        # st.write(movie[1])
        # image = Image.open(movie[1][0])
    # st.write(posters[0])
    # col1 = st.columns(1)
    # with col1:
    #     st.header(names[0])
    #     st.image(posters[0])

    # image = Image.open(r'{}'.format(posters[0]))
    # st.image(image)

