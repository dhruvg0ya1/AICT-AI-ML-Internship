import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import requests
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.user_movie_matrix = None
        self.svd_model = None
        self.movie_features = None
        
    def download_movielens_data(self):
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        
        if not os.path.exists('ml-latest-small'):
            st.info("Downloading MovieLens dataset... This may take a moment.")
            try:
                response = requests.get(url)
                with open('ml-latest-small.zip', 'wb') as f:
                    f.write(response.content)
                
                with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                os.remove('ml-latest-small.zip')
                st.success("Dataset downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading dataset: {e}")
                return False
        return True
    
    def load_data(self):
        if not self.download_movielens_data():
            return False
            
        try:
            self.movies = pd.read_csv('ml-latest-small/movies.csv')
            self.ratings = pd.read_csv('ml-latest-small/ratings.csv')
            
            self.movies['genres'] = self.movies['genres'].str.replace('|', ' ')
            title_year_extract = self.movies['title'].str.extract(r'(.*) \((\d{4})\)')
            self.movies['clean_title'] = title_year_extract[0].fillna(self.movies['title'])
            self.movies['year'] = title_year_extract[1].fillna('Unknown')
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def prepare_content_based_features(self):
        self.movie_features = self.movies['genres'].fillna('') + ' ' + self.movies['clean_title'].fillna('')
        
        tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movie_features)
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
    
    def content_based_recommend(self, movie_title, n_recommendations=5):
        movie_matches = self.movies[self.movies['clean_title'].str.contains(movie_title, case=False, na=False)]
        
        if movie_matches.empty:
            return None, "Movie not found in database"
        
        movie_idx = movie_matches.index[0]
        
        sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        
        recommended_movies = self.movies.iloc[movie_indices][['title', 'genres']].copy()
        recommended_movies['similarity_score'] = [sim_scores[i+1][1] for i in range(n_recommendations)]
        
        return recommended_movies, None
    
    def prepare_collaborative_filtering(self):
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        user_movie_sparse = csr_matrix(self.user_movie_matrix.values)
        
        # SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        user_factors = self.svd_model.fit_transform(user_movie_sparse)
        
        self.user_factors = user_factors
        self.movie_factors = self.svd_model.components_.T
    
    def collaborative_filtering_recommend(self, user_id, n_recommendations=5):
        if user_id not in self.user_movie_matrix.index:
            return None, "User not found in database"
        
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Calculate predicted ratings using SVD
        user_vector = self.user_factors[user_idx]
        predicted_ratings = np.dot(user_vector, self.movie_factors.T)
        
        unrated_movies = user_ratings[user_ratings == 0].index
        
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.user_movie_matrix.columns)}
        unrated_predictions = []
        
        for movie_id in unrated_movies:
            if movie_id in movie_id_to_idx:
                movie_idx = movie_id_to_idx[movie_id]
                predicted_rating = predicted_ratings[movie_idx]
                unrated_predictions.append((movie_id, predicted_rating))
        
        unrated_predictions.sort(key=lambda x: x[1], reverse=True)
        
        top_movie_ids = [movie_id for movie_id, rating in unrated_predictions[:n_recommendations]]
        
        recommended_movies = self.movies[self.movies['movieId'].isin(top_movie_ids)][['title', 'genres']].copy()
        recommended_movies['predicted_rating'] = [rating for movie_id, rating in unrated_predictions[:n_recommendations]]
        
        return recommended_movies, None
    
    def get_popular_movies(self, n_movies=10):
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        
        movie_stats.columns = ['avg_rating', 'rating_count']
        
        # Filter movies with at least 50 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 50]
        popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
        
        top_movie_ids = popular_movies.head(n_movies).index
        popular_movies_details = self.movies[self.movies['movieId'].isin(top_movie_ids)][['title', 'genres']].copy()
        
        return popular_movies_details

def main():
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Discover your next favorite movie!")
    
    if 'recommender' not in st.session_state:
        st.session_state.recommender = MovieRecommendationSystem()
        st.session_state.data_loaded = False
    
    recommender = st.session_state.recommender
    
    # Auto-load data on first run
    if not st.session_state.data_loaded:
        with st.spinner("Loading data... This may take a moment."):
            if recommender.load_data():
                recommender.prepare_content_based_features()
                recommender.prepare_collaborative_filtering()
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                st.rerun()
    
    if st.session_state.data_loaded:
        st.sidebar.title("Navigation")
        option = st.sidebar.selectbox(
            "Choose Recommendation Type",
            ["Content-Based Filtering", "Collaborative Filtering", "Popular Movies"]
        )
        
        if option == "Content-Based Filtering":
            st.header("Content-Based Recommendations")
            st.markdown("Get recommendations based on movie genres and content similarity.")
            
            movie_title = st.text_input("Enter a movie title:")
            
            if st.button("Get Recommendations") and movie_title:
                with st.spinner("Finding similar movies..."):
                    recommendations, error = recommender.content_based_recommend(movie_title)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Top 5 movies similar to '{movie_title}':")
                        
                        for idx, row in recommendations.iterrows():
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"*Genres:* {row['genres']}")
                            st.markdown(f"*Similarity Score:* {row['similarity_score']:.3f}")
                            st.markdown("---")
        
        elif option == "Collaborative Filtering":
            st.header("Collaborative Filtering Recommendations")
            st.markdown("Get recommendations based on users with similar preferences.")
            
            user_id = st.number_input("Enter User ID:", min_value=1, max_value=610, value=1)
            
            if st.button("Get Recommendations"):
                with st.spinner("Analyzing user preferences..."):
                    recommendations, error = recommender.collaborative_filtering_recommend(user_id)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Top 5 movie recommendations for User {user_id}:")
                        
                        for idx, row in recommendations.iterrows():
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"*Genres:* {row['genres']}")
                            st.markdown(f"*Predicted Rating:* {row['predicted_rating']:.2f}")
                            st.markdown("---")
        
        else:
            st.header("Popular Movies")
            st.markdown("Discover highly-rated movies with many reviews.")
            
            if st.button("Show Popular Movies"):
                popular_movies = recommender.get_popular_movies()
                
                st.success("Top 10 Popular Movies:")
                for idx, row in popular_movies.iterrows():
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"*Genres:* {row['genres']}")
                    st.markdown("---")
        
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.markdown(f"**Movies:** {len(recommender.movies)}")
        st.sidebar.markdown(f"**Users:** {recommender.ratings['userId'].nunique()}")
        st.sidebar.markdown(f"**Ratings:** {len(recommender.ratings)}")
        
        st.sidebar.markdown("### Sample Movies")
        sample_movies = recommender.movies.sample(5)['title'].tolist()
        for movie in sample_movies:
            st.sidebar.markdown(f"- {movie}")
    
    else:
        st.markdown("""
        ## About this Movie Recommendation System
        
        This system implements two main recommendation approaches:
        
        ### 🎯 Content-Based Filtering
        - Recommends movies based on genres and content similarity
        - Uses TF-IDF vectorization and cosine similarity
        - Great for discovering movies similar to ones you already like
        
        ### 👥 Collaborative Filtering
        - Recommends movies based on user behavior patterns
        - Uses matrix factorization (SVD) to find similar users
        - Discovers movies liked by users with similar tastes
        
        ### 🔥 Popular Movies
        - Shows highly-rated movies with many reviews
        - Good starting point for new users
        
        **Dataset:** MovieLens Latest Small Dataset (~100k ratings)
        """)

if __name__ == "__main__":
    main()
