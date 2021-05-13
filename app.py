from flask import Flask, render_template, redirect, url_for
from form import RecommendForm


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

app.config["SECRET_KEY"] = "5791628bb0b13ce0c676dfde280ba245"

#### helper funciton #####


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
#####


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_homepage_from_index(index):
    return df[df.index == index]["homepage"].values[0]


def get_average_vote_from_index(index):
    return df[df.index == index]["average_vote"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


##################################################


# Step 1: Read CSV File
df = pd.read_csv("C:/Enviroments/flaskblog/flaskblog/movie_dataset.csv")

# Step 2: Select Features

features = ["keywords", "cast", "genres", "director"]

# Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna("")


def combine_features(row):
    try:
        return (
            row["keywords"]
            + " "
            + row["cast"]
            + " "
            + row["genres"]
            + " "
            + row["director"]
        )
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)

# Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)


@app.route("/recommendations/<name>")
def recommendations(name):
    # print(name)
    movie_index = get_index_from_title(name)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)
    movies = []
    for element in range(0, 16, 1):
        movies.append(get_title_from_index(sorted_similar_movies[element][0]))

    return render_template("rec.html", movies=movies,)


@app.route("/movies", methods=["GET", "POST"])
def movies():
    form = RecommendForm()
    if form.validate_on_submit():
        if form.validate_on_submit():
            return redirect(url_for("recommendations", name=form.name.data))

    return render_template("movies.html", form=form)


@app.route("/")
@app.route("/home")
def home():
    return render_template("layout.html")


if __name__ == "__main__":
    app.run(debug=True)
