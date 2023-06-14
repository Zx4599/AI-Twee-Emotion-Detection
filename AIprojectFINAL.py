import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
# Load the dataset
positive_tweets = pd.read_csv("Postive Tweet.tsv", sep="\t", header=None, names=["tweet"])
negative_tweets = pd.read_csv("Negtive Tweet.tsv", sep="\t", header=None, names=["tweet"])

# Add label column to the dataset
positive_tweets["label"] = "positive"
negative_tweets["label"] = "negative"

# Combine the positive and negative tweets into one dataset
tweets = pd.concat([positive_tweets, negative_tweets], ignore_index=True)

# Preprocess the tweets text
def preprocess_tweet(tweet):
    # Remove punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenize the words
    words = nltk.word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words("arabic"))
    words = [word for word in words if word not in stop_words]

    # Stem the words
    stemmer = SnowballStemmer("arabic")
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    tweet = " ".join(words)

    return tweet

# Apply preprocessing to the tweets
tweets["tweet"] = tweets["tweet"].apply(preprocess_tweet)

# Extract features from the tweets using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(tweets["tweet"])

# Split the dataset into training and testing sets (75% training and 25% testing)
X_train, X_test, y_train, y_test = train_test_split(features, tweets["label"], test_size=0.25)

# Train a Naive Bayes classifier using the classical method
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

# Evaluate the Naive Bayes classifier using the classical method
nb_clf_score = nb_clf.score(X_test, y_test)
print("Naive Bayes Classical Method Test Set Accuracy: %0.2f" % nb_clf_score)

# Train a Decision Trees classifier using the classical method
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Evaluate the Decision Trees classifier using the classical method
dt_clf_score = dt_clf.score(X_test, y_test)
print("Decision Trees Classical Method Test Set Accuracy: %0.2f" % dt_clf_score)

# Train a Naive Bayes classifier using 5-fold cross-validation
nb_cv_clf = MultinomialNB()
nb_cv_scores = cross_val_score(nb_cv_clf, features, tweets["label"], cv=5)
print("Naive Bayes 5-Fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (nb_cv_scores.mean(), nb_cv_scores.std() * 2))

def read_tweets_from_file(filename):
    tweets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            preprocessed_tweet = preprocess_tweet   (line.strip())
            tweets.append(preprocessed_tweet)
    return tweets

# Train a Decision Trees classifier using 5-fold cross-validation
dt_cv_clf = DecisionTreeClassifier()
dt_cv_scores = cross_val_score(dt_cv_clf, features, tweets["label"], cv=5)
print("Decision Trees 5-Fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (dt_cv_scores.mean(), dt_cv_scores.std() * 2))
print('-----------------')
# Read tweets from file
tweets = read_tweets_from_file("tweets.txt")

# Extract features from the tweets using TF-IDF
features = vectorizer.transform(tweets)

# Predict the sentiment of the tweets using the trained classifiers
nb_predictions = nb_clf.predict(features)
dt_predictions = dt_clf.predict(features)

# Print the predictions
for i in range(len(tweets)):
    print("Tweet:", tweets[i])
    print("Naive Bayes classifier prediction:", nb_predictions[i])
    print("Decision Trees classifier prediction:", dt_predictions[i])
    print("--------")
