from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the user data
users_df = pd.read_csv('BX-Users.csv', delimiter=';', encoding='ISO-8859-1')

# Clean the data (optional, based on your needs)
users_df['Age'] = users_df['Age'].fillna('Unknown')

@app.route('/')
def index():
    # Fetch all users data for the homepage
    users = users_df.to_dict(orient='records')
    return render_template('index.html', users=users)

@app.route('/filter', methods=['GET', 'POST'])
def filter_users():
    # Handle filtering logic
    age_filter = request.form.get('age')
    location_filter = request.form.get('location')

    filtered_df = users_df.copy()

    if age_filter:
        filtered_df = filtered_df[filtered_df['Age'] == int(age_filter)]
    
    if location_filter:
        filtered_df = filtered_df[filtered_df['Location'].str.contains(location_filter, case=False)]

    users = filtered_df.to_dict(orient='records')
    return render_template('index.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)
