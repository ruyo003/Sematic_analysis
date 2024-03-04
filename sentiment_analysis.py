# this script performs a sentiment analysis for the text reviews on amazon products

import spacy
import pandas as pd
from textblob import TextBlob


nlp = spacy.load('en_core_web_sm')

# reading the amazon dataset
amazon_df = pd.read_csv('Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

# accessing the text review column in the dataset and dropping null values in the column
clean_reviews = amazon_df.dropna(subset = ['reviews.text'], inplace = True)

text_reviews = amazon_df['reviews.text']
#print(text_reviews)

# checking for null values in the data set including the column we are interested in
missing_values = amazon_df.isnull().sum()
#print(missing_values)

# the function below takes a review, cleans the text and predicts the sentiments.
def sentiment_analysis(review):
    # process of cleaning the review, starting by changing everything to lower case

    first_review = nlp(review. lower())
    print(first_review)
    # removing stop words in the text reviews
    # iterating through the review to get the stop words
    stop_words =[word for word in first_review if word.is_stop == True]

    #print(stop_words)

    review_without_stopwords = [word for word in first_review if word not in stop_words]

    #print(review_without_stopwords)

    # tokenizing the review, removing any punctuation marks and whitespaces
    token_review = ([token.orth_ for token in review_without_stopwords if not token.is_punct | token.is_space])
    # print(f'This is the tokenized data: {token_review}')

    joint_review = ', '.join(token_review)
    # print(joint_review)

    # determining the sentiment and the polarity score of the review
    blob_review = TextBlob(joint_review)
    polarity_score = blob_review.sentiment.polarity

    # categorising the score into negative, positive or neutral
    if polarity_score < 0:
        print('The sentiment analysis is: negative')
    elif polarity_score > 0:
        print('The sentiment analysis is: positive')
    else:
        print('The sentiment analysis is: neutral')

    return polarity_score

# testing the model using the reviews in the dataset

first = sentiment_analysis(text_reviews[27])
second = sentiment_analysis(text_reviews[55])
third = sentiment_analysis(text_reviews[1002])


print('the sentiment is', first)
print('the sentiment is', second)
print('the sentiment is', third)


