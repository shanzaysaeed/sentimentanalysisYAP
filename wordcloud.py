import pandas as pd
from textblob import TextBlob

df = pd.read_csv('final_data.csv')

reviews = df["Review"].dropna()

reviews = reviews.str.lower()
reviews = reviews.str.split(" ")
review_words = list(reviews)

review_words = [item for sublist in review_words for item in sublist]

unique_words = list(set(review_words))
unique_words.remove("")

label = ["Word", "Sentiment"]
word_df = pd.DataFrame(columns=label)
word_df["Word"] = unique_words

for i in word_df.iloc(0):
    word = i["Word"]
    blob = TextBlob(word)
    s = blob.sentiment
    if s.polarity > 0:
        i["Sentiment"] = "positive"
    elif s.polarity < 0:
        i["Sentiment"] = "negative"
    else:
        i["Sentiment"] = "neutral"

word_df.drop(word_df.index[word_df['Sentiment'] == "neutral"], inplace=True)

word_df = word_df.reset_index(drop=True)
print (word_df)

word_df.to_csv("wordcloud.csv")
