from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib
import os
import logging


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Load the user and ratings data
users_df = pd.read_csv('data/BX-Users.csv', delimiter=';', encoding='ISO-8859-1')
books_df = pd.read_csv('data/BX-Books.csv', delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')
ratings_df = pd.read_csv('data/BX-Book-Ratings.csv', delimiter=';', encoding='ISO-8859-1')

# Clean the data (optional)
users_df['Age'] = users_df['Age'].fillna('Unknown')

def save_data():
    try:
        users_df.to_csv('uploads/BX-Users.csv', index=False, sep=';', encoding='ISO-8859-1')
        ratings_df.to_csv('uploads/BX-Book-Ratings.csv', index=False, sep=';', encoding='ISO-8859-1')
        books_df.to_csv('uploads/BX-Books.csv', index=False, sep=';', encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error saving data: {e}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users')
def users():
    users = users_df.to_dict(orient='records')
    return render_template('users.html', users=users)

@app.route('/filter', methods=['GET', 'POST'])
def filter_users():
    age_filter = request.form.get('age')
    location_filter = request.form.get('location')

    filtered_df = users_df.copy()

    if age_filter:
        filtered_df = filtered_df[filtered_df['Age'] == int(age_filter)]
    
    if location_filter:
        filtered_df = filtered_df[filtered_df['Location'].str.contains(location_filter, case=False)]

    users = filtered_df.to_dict(orient='records')
    return render_template('filters_users.html', users=users)

@app.route('/books')
def books():
    books = books_df.to_dict(orient='records')
    return render_template('books.html', books=books)

@app.route('/book/<isbn>')
def book_detail(isbn):
    book = books_df[books_df['ISBN'] == isbn].to_dict(orient='records')
    if book:
        return render_template('book_detail.html', book=book[0])
    else:
        return "Book not found", 404

@app.route('/books/filter', methods=['GET', 'POST'])
def filter_books():
    min_rating = request.form.get('min_rating')

    filtered_df = ratings_df.copy()

    if min_rating:
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean().reset_index()
        avg_ratings = avg_ratings[avg_ratings['Book-Rating'] >= float(min_rating)]
        books = avg_ratings.to_dict(orient='records')
    else:
        books = []

    return render_template('books.html', books=books)

@app.route('/book/edit/<isbn>', methods=['GET', 'POST'])
def edit_book(isbn):
    book = books_df[books_df['ISBN'] == isbn]
    if request.method == 'POST':
        # Get data from the form
        title = request.form['title']
        author = request.form['author']
        year = request.form['year']
        publisher = request.form['publisher']

        # Update the book details
        books_df.loc[books_df['ISBN'] == isbn, ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']] = [title, author, year, publisher]

         # Save the updated book details to the CSV file
        save_data()
        
        # Redirect to the updated book's detail page
        return redirect(url_for('book_detail', isbn=isbn))

    if book.empty:
        return "Book not found", 404
    return render_template('edit_book.html', book=book.iloc[0])

@app.route('/ratings')
def ratings():
    return render_template('ratings.html')

@app.route('/ratings/most-rated')
def most_rated_books():
    most_rated = ratings_df.groupby('ISBN').size().reset_index(name='Counts')
    most_rated = most_rated.sort_values(by='Counts', ascending=False).head(10)
    most_rated_books = most_rated.to_dict(orient='records')
    return render_template('most_rated.html', books=most_rated_books)

@app.route('/ratings/distribution')
def rating_distribution():
    img = io.BytesIO()

    plt.figure(figsize=(10, 6))
    ratings_df['Book-Rating'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Book Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('distribution.html', plot_url=plot_url)

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    search_results = []

    if query:
        # Search for matching users
        user_matches = users_df[users_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        search_results.extend(user_matches.to_dict(orient='records'))   
        
        # Search for matching books based on ISBN
        book_matches = ratings_df[ratings_df['ISBN'].str.contains(query, case=False, na=False)]
        book_results = book_matches.groupby('ISBN')['Book-Rating'].mean().reset_index()
        book_results.columns = ['ISBN', 'Average Rating']
        search_results.extend(book_results.to_dict(orient='records'))

    return render_template('search_results.html', query=query, results=search_results)


# Global variables to hold the data and model paths
book_features_df_matrix = None
model_knn_path = 'knn_model.pkl'
book_features_df = None



# Function to load data
def load_data(books_filename, ratings_filename):
    # Load books data
    df_books = pd.read_csv(
        books_filename,
        encoding="ISO-8859-1",
        sep=";",
        header=0,
        names=['bookId', 'title', 'author'],
        usecols=['bookId', 'title', 'author'],
        dtype={'bookId': 'str', 'title': 'str', 'author': 'str'}
    )

    # Load ratings data
    df_ratings = pd.read_csv(
        ratings_filename,
        encoding="ISO-8859-1",
        sep=";",
        header=0,
        names=['userId', 'bookId', 'rating'],
        usecols=['userId', 'bookId', 'rating'],
        dtype={'userId': 'int32', 'bookId': 'str', 'rating': 'float32'}
    )

    # Return both DataFrames
    return df_books, df_ratings

# Function to train the model
def train_model(df_books, df_ratings):
    # Get list of users to remove
    user_rating_count = (df_ratings
                         .groupby(by=['userId'])['rating']
                         .count()
                         .reset_index()
                         .rename(columns={'rating': 'totalRatingCount'})
                         [['userId', 'totalRatingCount']]
                         )
    users_to_remove = user_rating_count.query('totalRatingCount > 200').userId.tolist()

    # Merge rating and catalog by bookId
    df = pd.merge(df_ratings, df_books, on='bookId')

    # Create totalRatingCount for books
    book_rating_count = (df
                         .groupby(by=['title'])['rating']
                         .count()
                         .reset_index()
                         .rename(columns={'rating': 'totalRatingCount'})
                         [['title', 'totalRatingCount']]
                         )

    rating_with_total_rating_count = df.merge(book_rating_count, left_on='title', right_on='title', how='left')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Remove books with less than 100 ratings
    rating_popular_movie = rating_with_total_rating_count.query('totalRatingCount > 100')

    # Remove users with less than 200 ratings
    rating_popular_movie = rating_popular_movie[rating_popular_movie['userId'].isin(users_to_remove)]

    # Pivot table and create matrix
    book_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    book_features_df_matrix = csr_matrix(book_features_df.values)

    # Train the k-NN model
    model_knn = NearestNeighbors(metric='cosine', n_neighbors=5, algorithm='auto')
    model_knn.fit(book_features_df_matrix)

    # Return the model and the book features DataFrame
    return model_knn, book_features_df

@app.route('/train_model', methods=['GET', 'POST'])
def train_model_route():
    if request.method == 'POST':
        try:
            # Load data
            df_books, df_ratings = load_data('data/BX-Books.csv', 'data/BX-Book-Ratings.csv')

            # Train the model
            model_knn, book_features_df = train_model(df_books, df_ratings)

            status = "Model trained successfully!"
            return render_template('training.html', status=status)
        except Exception as e:
            error = f"An error occurred during training: {str(e)}"
            return render_template('training.html', error=error)
    return render_template('training.html')

# Function to recommend books based on a specific book title
def recommend(model_knn, book_features_df, book):
    # Find the index of the book in the DataFrame
    query_index = None
    for idx in range(len(book_features_df)):
        if book_features_df.index[idx] == book:
            query_index = idx
            break

    if query_index is None:
        return [book, []]  # Book not found

    # Create the return structure
    ret = [book_features_df.index[query_index], []]
    distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index, :].values.reshape(1, -1))

    # Now we located the book. Let's show the recommendations
    for i in range(1, len(distances.flatten())):
        ret[1].insert(0, [book_features_df.index[indices.flatten()[i]], distances.flatten()[i]])

    return ret

# Function to get recommendations for a specific book
def get_recommends(book=""):
    # Load data
    df_books, df_ratings = load_data('data/BX-Books.csv', 'data/BX-Book-Ratings.csv')

    # Train model
    model_knn, book_features_df = train_model(df_books, df_ratings)

    # Get recommendations
    recommendations = recommend(model_knn, book_features_df, book)
    
    return recommendations

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_route():
    if request.method == 'POST':
        book_title = request.form.get('bookTitle')
        try:
            recommendations = get_recommends(book_title)
            if recommendations:
                return render_template('recommend.html', recommendations=recommendations[1], bookTitle=recommendations[0])
            else:
                error = "No recommendations found. Please check the book title."
                return render_template('recommend.html', error=error)
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            return render_template('recommend.html', error=error)
    return render_template('recommend.html')


# Example test function
def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False

    print(recommends)
    print()

    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]

    print(recommended_books)
    print(recommended_books_dist)

    for i in range(2):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")



if __name__ == '__main__':
    app.run(debug=True)
