# movie-recommender-cf

Movie Recommender System
This project is a movie recommendation system built using different machine learning techniques on a real-world ratings dataset.
The goal was to understand how recommendation algorithms work in practice and compare their performance.

Models Used

In this project, I implemented:
 • Item-Based Collaborative Filtering with cosine similarity and rating normalization.
 • Matrix Factorization (Truncated SVD) to learn latent user–movie patterns.
 • A simple Content-Based approach based on movie genres.

All models were evaluated using RMSE to compare their performance.
What I Focused On
 • Proper train/test split
 • Rating normalization (mean-centering)
 • Clean modular project structure
 • Comparing different recommendation strategies
 • Basic optimization of model parameters

 Tech Stack
 • Python
 • Pandas
 • NumPy
 • Scikit-learn

 How To Run:
 
pip install -r requirements.txt
python main.py
