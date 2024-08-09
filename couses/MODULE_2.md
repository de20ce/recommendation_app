
# Module 2: Collaborative Filtering

---

## Slide 1: Introduction to Collaborative Filtering

### What is Collaborative Filtering?

- **Collaborative Filtering** is a technique used by recommendation systems to predict a user's interest by collecting preferences from many users (collaborating). It assumes that if a user A has the same opinion as user B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen user.

### Types of Collaborative Filtering
- **User-Based Collaborative Filtering**: Recommendations based on the similarities between users.
- **Item-Based Collaborative Filtering**: Recommendations based on the similarities between items.

---

## Slide 2: User-Based Collaborative Filtering

### Concept and Workflow

- **User Similarity**:
  - Calculate the similarity between users based on their ratings of items.
  - **Similarity Measures**: Cosine similarity, Pearson correlation, Euclidean distance.

- **Recommendation Process**:
  - Identify a user’s nearest neighbors (users with similar preferences).
  - Recommend items that these similar users have rated highly but the target user has not yet interacted with.

### Example
- **Scenario**: If User A and User B both liked the same set of movies, and User B liked a movie that User A hasn't seen, User A might be recommended that movie.

**Citations**:
- "User-based collaborative filtering is one of the earliest and most widely used techniques in recommender systems." [Resnick et al., 1994]

---

## Slide 3: Item-Based Collaborative Filtering

### Concept and Workflow

- **Item Similarity**:
  - Calculate the similarity between items based on how users have rated them.
  - **Similarity Measures**: Cosine similarity, Pearson correlation.

- **Recommendation Process**:
  - Identify items similar to those the user has already rated highly.
  - Recommend similar items based on the target user's previous ratings.

### Example
- **Scenario**: If a user liked a particular book, the system recommends other books that are similar in content, genre, or user ratings.

**Citations**:
- "Item-based collaborative filtering is often preferred for its stability and scalability, especially in large datasets." [Sarwar et al., 2001]

---

## Slide 4: Matrix Factorization Techniques

### Introduction to Matrix Factorization

- **Matrix Factorization**:
  - Decomposes a large user-item interaction matrix into lower-dimensional matrices that capture latent factors.
  - Useful for capturing the hidden patterns in user preferences and item characteristics.

### Key Techniques
- **Singular Value Decomposition (SVD)**:
  - Factorizes the matrix into three components: user preferences, item characteristics, and singular values.
  - Effective for reducing dimensionality and improving recommendation accuracy.

- **Alternating Least Squares (ALS)**:
  - An iterative optimization algorithm often used in collaborative filtering for large-scale datasets, particularly with implicit feedback.

**Citations**:
- "Matrix factorization techniques have become the dominant approach for collaborative filtering since the Netflix Prize competition." [Koren et al., 2009]

---

## Slide 5: Non-negative Matrix Factorization (NMF)

### Concept and Application

- **Non-negative Matrix Factorization (NMF)**:
  - Similar to SVD, but with non-negativity constraints on the factorized matrices.
  - Helps in interpretability by ensuring all latent factors are non-negative, which can be more intuitive (e.g., positive influences only).

### Example Application
- **Recommending Movies**:
  - NMF can be used to decompose a user-movie rating matrix into latent factors representing different genres or movie features that influence user ratings.

**Pros**:
- Better interpretability due to non-negative factors.
- Suitable for applications where features are inherently non-negative (e.g., ratings, user interactions).

**Citations**:
- "NMF provides a more interpretable factorization compared to SVD, making it useful in scenarios where positive latent factors are desired." [Lee and Seung, 1999]

---

## Slide 6: Practical Implementation of Collaborative Filtering

### Implementing User-Based Filtering in Python

- **Steps**:
  1. Load user-item interaction data.
  2. Compute similarity between users using a similarity measure (e.g., cosine similarity).
  3. Identify the nearest neighbors for each user.
  4. Predict ratings for items the user has not yet interacted with.
  5. Recommend top-N items based on predicted ratings.

### Code Example
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assume user_item_matrix is a matrix with users as rows and items as columns
similarity_matrix = cosine_similarity(user_item_matrix)

# Function to predict ratings
def predict_ratings(user_id, item_id):
    user_similarity = similarity_matrix[user_id]
    user_ratings = user_item_matrix[:, item_id]
    return np.dot(user_similarity, user_ratings) / np.sum(user_similarity)
```

### Implementing Item-Based Filtering in Python

- **Steps**:
  1. Transpose the user-item matrix to item-user.
  2. Compute similarity between items.
  3. Recommend items similar to those the user has rated highly.

**Citations**:
- "Practical implementation of collaborative filtering can be achieved through various libraries in Python, such as Scikit-Learn and Surprise." [Hug, 2020]

---

## Slide 7: Evaluation and Challenges in Collaborative Filtering

### Evaluation Metrics
- **Precision and Recall**: Measure the accuracy of the recommendations.
- **Mean Squared Error (MSE)**: Measures the difference between predicted and actual ratings.
- **Coverage**: The percentage of items or users for which the system can provide recommendations.

### Challenges
- **Cold Start Problem**: Difficulty in recommending items to new users with no interaction history.
- **Data Sparsity**: Many recommendation matrices are sparse, leading to unreliable similarity calculations.
- **Scalability**: Efficiently handling large-scale datasets with millions of users and items.

**Citations**:
- "Challenges such as cold start and data sparsity are significant in collaborative filtering, necessitating the use of hybrid methods and advanced algorithms." [Schein et al., 2002]

---

## Learning Objectives Recap

- **Understand** the principles of user-based and item-based collaborative filtering.
- **Learn** how to implement collaborative filtering algorithms using Python.
- **Explore** advanced matrix factorization techniques for improving recommendation accuracy.
- **Evaluate** the effectiveness of collaborative filtering models using appropriate metrics.
- **Identify** the challenges and limitations of collaborative filtering and how to address them.

---

### Additional Resources

- **Books**:
  - "Programming Collective Intelligence" by Toby Segaran.
  - "Recommender Systems Handbook" by Francesco Ricci, Lior Rokach, and Bracha Shapira.

- **Research Papers**:
  - Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). GroupLens: An Open Architecture for Collaborative Filtering of Netnews. In CSCW '94.
  - Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-Based Collaborative Filtering Recommendation Algorithms. In WWW '01.
  - Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. In IEEE Computer.

- **Online Courses**:
  - Coursera: "Machine Learning" by Stanford University (focus on collaborative filtering).
  - DataCamp: "Building Recommendation Engines with PySpark."
