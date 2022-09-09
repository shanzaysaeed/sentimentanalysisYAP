#THIS FILE CONTAINS COMPLETE SENTIMENT ANALYSIS TOOL.
# newpush
# If you want to turn off or on a specific scraper read the following instruction:
# Write the names of the scrapers you want to turn ON in front of scraper-list
# use the names exactly as follows:

# FOR "FACEBOOK" WRITE: facebook
# FOR "GOOGLE STORE" WRITE: playstore
# FOR "APPLE STORE" WRITE: appstore
# FOR "TWITTER" WRITE: twitter
# FOR "INSTAGRAM" WRITE: instagram

# write the following comma seperated in inverted columns between the [] below:
# for example scraper_list = ["playstore" , "appstore"] 

# scraper_list = ["playstore" , "appstore" ,"twitter" , "instagram", "facebook"]
scraper_list = [ "instagram"]
instagram_pages_list = ["yap",  "yappakistan", "yapuae"]


# ❌❌❌❌❌❌DONOT EDIT THE CODE BELOW ❌❌❌❌❌

# Useful Libraries 
from google_play_scraper import app
import instaloader
from google_play_scraper import Sort, reviews_all
from app_store_scraper import AppStore
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from facebook_scraper import get_posts
from datetime import date
import time
import datetime
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from googletrans import Translator
from tracemalloc import stop
from tracemalloc import stop
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding


def playstore_scrapper():
    # YAP Google app reviews extractor
    yap_reviews = reviews_all(
        'com.yap.banking',
        sleep_milliseconds=0, # defaults to 0
        lang='en', # defaults to 'en'
        country='us', # defaults to 'us'
        sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    )

    # Making a data frame
    df_reviews = pd.DataFrame(np.array(yap_reviews),columns=['review'])
    df_reviews = df_reviews.join(pd.DataFrame(df_reviews.pop('review').tolist()))

    # Data cleaning 
    spec_chars = ["!",'"',"#","%","&","'","(",")",
                "*","+",",","-",".","/",":",";","<",
                "=",">","?","@","[","\\","]","^","_",
                "`","{","|","}","~","–"]

    for char in spec_chars:
        df_reviews['content'] = df_reviews['content'].str.replace(char, ' ')

    df_reviews['content'] = df_reviews['content'].str.split().str.join(" ")

    df_reviews['content'] = df_reviews['content'].str.replace('[^A-Za-z0-9 ]', '')

    df_reviews = df_reviews.dropna(subset=['content'])

    # Converting and Saving Dataframe as CSV
    df_reviews.to_csv('GoogleStore.csv')

def appstore_scrapper():
    # Extracting Reviews from AppStore using App_Store_Scraper Library
    yap_reviews = AppStore(country="us", app_name="yap-your-digital-banking-app", app_id='1498302242')
    yap_reviews.review()

    # Making Pandas DataFrame
    df_reviews = pd.DataFrame(np.array(yap_reviews.reviews),columns=['review'])
    df_reviews = df_reviews.join(pd.DataFrame(df_reviews.pop('review').tolist()))

    # Data Cleaning
    spec_chars = ["!",'"',"#","%","&","'","(",")",
                "*","+",",","-",".","/",":",";","<",
                "=",">","?","@","[","\\","]","^","_",
                "`","{","|","}","~","–"]

    for char in spec_chars:
        df_reviews['review'] = df_reviews['review'].str.replace(char, ' ')

    df_reviews['review'] = df_reviews['review'].str.split().str.join(" ")

    df_reviews['review'] = df_reviews['review'].str.replace('[^A-Za-z0-9 ]', '')

    df_reviews = df_reviews.dropna(subset=['review'])

    # Converting and Saving Dataframe as CSV
    df_reviews.to_csv('AppStore.csv')

def twitter_scrapper():
    print ("\n\nExtracting Tweets\n")
    # Creating list to append tweet data to
    tweets_list = []

    # Getting today's date from system
    today_date = str(date.today())
    # Start date for tweets (YAP twitter account was made in 2012)
    start_date = str(date(2019,1,1))
    # No of tweets to search
    n = 1000000

    # Search Query for searching tweets (I used keyword 'YAP digital bank')
    search = 'YAP digital bank'
    search_query = search + ' since:' + start_date + ' until:' + today_date

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        if i > n:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username])
        
    # Creating a dataframe from the tweets list above
    df_label = ['Datetime', 'Tweet Id', 'Text', 'Username']
    tweets_df = pd.DataFrame(tweets_list, columns=df_label)


    # Cleaning the dataset
    spec_chars = ["!",'"',"#","%","&","'","(",")",
                "*","+",",","-",".","/",":",";","<",
                "=",">","?","@","[","\\","]","^","_",
                "`","{","|","}","~","–"]

    for char in spec_chars:
        tweets_df['Text'] = tweets_df['Text'].str.replace(char, ' ')

    tweets_df['Text'] = tweets_df['Text'].str.split().str.join(" ")

    tweets_df['Text'] = tweets_df['Text'].str.replace('[^A-Za-z0-9 ]', '')

    tweets_df = tweets_df.dropna(subset=['Text'])


    # Translating entire content column to english using Google_Trans Library
    # Initailizing Translator Object
    translator = Translator()

    # Translating entire Review Coloumn
    for i in range(len(tweets_df)):
        language = translator.detect(tweets_df["Text"][i])
        if language.lang != 'en':
            translated_text = translator.translate(tweets_df["Text"][i])
            tweets_df["Text"][i] = translated_text.text 

    # Converting and Saving Dataframe as CSV
    tweets_df.to_csv('Tweets.csv')

def facebook_scrapper():

    '''
    Optional parameters for facebook_scraper:

    cookies="cookies.json", 
    credentials=creds, 
    options={"comments": True},  
    extra_info=True, 
    '''

    # 2 ALTERNATIVES: 
    # # when scraping using Cookie File: (Uncomment the line below)
    # posts = get_posts("YAP", pages = 100, cookies = "cookies.json", options={"comments": True})
   
    # Else when Reading login creds from from file
    # When using credentails: (Uncomment the code below) 
    
    # login_file = open("fb_login.txt", "r")
    # login_data  = login_file.readline()
    # cred = login_data.split(" ")
    # email = str(cred[0])
    # password = str(cred[1])

    # creds = (str(email), str(password))

    # posts = get_posts("YAP", pages = 100, credentials=creds, options={"comments": True})

    # data_all = []
    # for post in posts:
    #     # print (post['post_id'] + " /get")
    #     for i in range (len(post['comments_full'])):
    #         data = []
    #         data.extend([post['post_id'], post['time'], post['username'], post['post_text'], post['post_url'], post['likes'], post['comments']])
    #         data.extend([post['comments_full'][i]['commenter_name'], post['comments_full'][i]['comment_text'], post['comments_full'][i]['comment_time']])
    #         data_all.append(data)

    # label = ["post_id", 'post_time', 'username', 'post_text', 'post_url', 'likes', 'comments', 'commenter_name', 'comment_text', 'comment_time']
    # new_df = pd.DataFrame(data_all, columns=label)

    # spec_chars = ["!",'"',"#","%","&","'","(",")",
    #             "*","+",",","-",".","/",":",";","<",
    #             "=",">","?","@","[","\\","]","^","_",
    #             "`","{","|","}","~","–"]

    # for char in spec_chars:
    #     new_df['comment_text'] = new_df['comment_text'].str.replace(char, ' ')

    # new_df['comment_text'] = new_df['comment_text'].str.split().str.join(" ")

    # new_df['comment_text'] = new_df['comment_text'].str.replace('[^A-Za-z0-9 ]', '')

    # new_df = new_df.dropna(subset=['comment_text'])

    # # Converting and Saving Dataframe as CSV
    # new_df.to_csv('Facebook_YAP.csv')
    print ("\nFacebook Scraper\n")

def instagram_scrapper():

    insta = instaloader.Instaloader()
    # Loading login session cookie file
    # Follow the link below to create one
    # https://instaloader.github.io/troubleshooting.html#login-error

    # insta.load_session_from_file("shanzaysaeed")

    login_file = open("insta_login.txt", "r")
    login_data  = login_file.readline()
    creds = login_data.split(" ")
    username = str(creds[0])
    password = str(creds[1])

    insta.login(username, password)

    # Enter Search Query ("yap", "yappakistan", "yapuae")

    def insta_comment(search_query):

        search_username = search_query


        # Extracting Posts and their Comments
        posts_data = []
        posts = instaloader.Profile.from_username(insta.context, search_username).get_posts()

        for post in posts:
            posturl = "https://www.instagram.com/p/" + post.shortcode
            # print("post date: "+str(post.date))
            # print("post profile: "+post.profile)
            # print("post caption: "+post.caption)
            # print("post url: " + posturl)
            
            print("post url: " + posturl)
            
            for comment in post.get_comments():
                data = []
                data.extend([post.mediaid, post.profile, post.caption, post.date, posturl, post.likes, post.comments])
                data.extend(([comment.id, comment.owner.username, comment.text, comment.created_at_utc]))
                posts_data.append(data)
                # print("comment username: "+comment.owner.username)
                # print("comment text: "+comment.text)
                # print("comment date : "+str(comment.created_at_utc))
            # print("\n")


        # Making Pandas DataFrame
        labels = ["Post_ID", "Post_Profile", "Post_Caption", "Post_Date", "Post_URL", "Post_Likes", "Post_Comments", "Comment_ID", "Comment_Username", "Comment_Text", "Comment_Time"]
        insta_df = pd.DataFrame(posts_data, columns=labels)


        # Data Cleaning
        spec_chars = ["!",'"',"#","%","&","'","(",")",
                    "*","+",",","-",".","/",":",";","<",
                    "=",">","?","@","[","\\","]","^","_",
                    "`","{","|","}","~","–"]

        for char in spec_chars:
            insta_df['Comment_Text'] = insta_df['Comment_Text'].str.replace(char, ' ')

        insta_df['Comment_Text'] = insta_df['Comment_Text'].str.split().str.join(" ")

        insta_df['Comment_Text'] = insta_df['Comment_Text'].str.replace('[^A-Za-z0-9 ]', '')
        insta_df = insta_df.dropna(subset=['Comment_Text'])
        file_name = "Instagram_" + search_query + ".csv"
        insta_df.to_csv(file_name)

    for pages in instagram_pages_list:
        insta_comment(pages)
        time.sleep(10)
   

def combined_scrappers():
    
    # Reading all data from CSVs
    # concating all data in seprate lists
    source = list()
    dates = list()
    username = list()
    review = list()

    for items in scraper_list:
        
        if items == "playstore":
            print ("\n\nExtracting PlayStore Reviews")
            playstore_scrapper()
            df_playstore = pd.read_csv("./GoogleStore.csv", index_col=0)
            df_playstore.insert(0, 'Source', 'playstore')
            source.append(list(df_playstore["Source"]))
            dates.append(list(df_playstore["at"]))
            username.append(list(df_playstore["userName"]))
            review.append(list(df_playstore["content"]))

        if items == "appstore":
            print ("\n\nExtracting AppStore Reviews")
            appstore_scrapper()
            df_appstore = pd.read_csv("./AppStore.csv", index_col=0)
            df_appstore.insert(0, 'Source', 'appstore')
            source.append(list(df_appstore["Source"]))
            dates.append(list(df_appstore["date"]))
            username.append(list(df_appstore["userName"]))
            review.append(list(df_appstore["review"]))

        if items == "twitter":
            twitter_scrapper()
            os.system('cls')
            df_tweets = pd.read_csv("./Tweets.csv", index_col=0)
            df_tweets.insert(0, 'Source', 'twitter')

            # Making Data Uniform
            date_time = list(df_tweets["Datetime"])
            new_date = []
            for i in range(len(date_time)):
                # print (i)
                year = date_time[i][0:4]
                month = date_time[i][5:7]
                date = date_time[i][8:10]
                time = date_time[i][10:16]

                # formatted_date = month + "/" + date + "/" + year + time
                formatted_date = datetime.datetime(int(year), int(month), int(date))
                new_date.append(formatted_date)
            df_tweets['Datetime'] = new_date

            source.append(list(df_tweets["Source"]))
            dates.append(list(df_tweets["Datetime"]))
            username.append(list(df_tweets["Username"]))
            review.append(list(df_tweets["Text"]))

        if items == "facebook":
            print ("\n\nExtracting Facebook Comments")
            # facebook_scrapper()
            # df_facebook = pd.read_csv("./Facebook_YAP.csv", index_col=0)
            # df_facebook.insert(0, 'Source', 'facebook')
            # source.append(list(df_facebook["Source"]))
            # dates.append(list(df_facebook["comment_time"]))
            # username.append(list(df_facebook["commenter_name"]))
            # review.append(list(df_facebook["comment_text"]))

        if items == "instagram":
            print ("\n\nExtracting Instagram Comments")
            instagram_scrapper()

            for i in range(len(instagram_pages_list)):
                filename = "./Instagram_" + instagram_pages_list[i] + ".csv" 
                dataframe = []
                dataframe.append( pd.read_csv(filename, index_col=0) )
            
            for i in range(len(dataframe)):
                dataframe[i].insert(0, 'Source', instagram_pages_list[i])

            for i in range(len(dataframe)):
                source.append(list(dataframe[i]["Source"]))
                dates.append(list(dataframe[i]["Comment_Time"]))
                username.append(list(dataframe[i]["Comment_Username"]))
                review.append(list(dataframe[i]["Comment_Text"]))

    # Flatening out the final list
    source = [item for sublist in source for item in sublist]
    dates = [item for sublist in dates for item in sublist]
    username = [item for sublist in username for item in sublist]
    review = [item for sublist in review for item in sublist]

    # here see the csv files for all scrapers and see the headings for source,date,comment,userid and added it accordingly
    labels = ["Source", "Datetime", "Username", "Review", "Region"]
    final_df = pd.DataFrame(columns=labels)

    # Inserting all data in the required column of final_df
    final_df["Source"] = source
    final_df["Datetime"] = dates
    final_df["Username"] = username
    final_df["Review"] = review

    final_df.to_csv("complete_data.csv")

def ml_model():
    
    # Reading data sets for training our model
    df_tweets_entire = pd.read_csv("./training_data.csv")
    df_roman_entire = pd.read_csv("./Roman Urdu DataSet.csv")
    use_cols=['Char','Neg','Neut','Pos']
    df_emojis_entire  = pd.read_csv("./emojis.csv", usecols=use_cols)

    # Copying the data sets by choosing specific columns
    df_tweets = df_tweets_entire[["text", "airline_sentiment"]].copy(deep=True)
    df_roman  = df_roman_entire[["Comment", "sentiment"]].copy(deep=True)

    # Renaming columns
    df_tweets.rename(columns = {'text':'Comment', 'airline_sentiment':'Sentiment'}, inplace = True)
    df_roman.rename(columns = {'sentiment':'Sentiment'}, inplace = True)

    # Renaming column values
    df_tweets["Sentiment"].replace({"Positive": "positive", "Negative": "negative"}, inplace=True)
    df_roman["Sentiment"].replace({"Positive": "positive", "Negative": "negative"}, inplace=True)

    # Converting Comment column to lower case for better accuracy
    df_tweets['Comment'] = df_tweets['Comment'].str.lower()
    df_roman['Comment'] = df_roman['Comment'].str.lower()

    # Removing neutral values 
    df_tweets = df_tweets[df_tweets['Sentiment'] != 'neutral']
    df_tweets = df_tweets[df_tweets['Sentiment'] != 'Neutral']
    df_roman = df_roman[df_roman['Sentiment'] != 'neutral']
    df_roman = df_roman[df_roman['Sentiment'] != 'Neutral']

    # Comparing positive and negative columns in emojis df
    comparison_column = np.where(df_emojis_entire["Pos"] >= df_emojis_entire["Neg"], True, False)
    df_emojis_entire['Sentiment'] = comparison_column
    df_emojis_entire["Sentiment"].replace({True: "positive", False: "negative"}, inplace=True)
    df_emojis_entire.rename(columns = {'Char':'Comment'}, inplace = True)
    df_emojis  = df_emojis_entire[["Comment", "Sentiment"]].copy(deep=True)
    df_emojis.drop(df_emojis.index[0], inplace=True)

    # Combining into one data frame
    frames = [df_tweets, df_roman, df_emojis]
    df = pd.concat(frames)

    # Adding rows in df for emoticon tagging
    row1 = {'Comment': 'happy', 'Sentiment': 'positive'}
    row2 = {'Comment': 'sad', 'Sentiment': 'negative'}
    df = df.append(row1, ignore_index = True)
    df = df.append(row2, ignore_index = True)
    df

    # factorizing the tags into 0 and 1 from Sentiment column
    factorized_sentiment = df.Sentiment.factorize()
    factorized_sentiment

    # Tokenization of the Comment column
    comments = df.Comment.values
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(comments)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(comments)
    padded_sequence = pad_sequences(encoded_docs, maxlen=200)
    
    # building the model
    embedding_vector_length = 32
    model = Sequential() 
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    print(model.summary()) 

    
    # running the model
    history = model.fit(padded_sequence, factorized_sentiment[0], validation_split=0.2, epochs=5, batch_size=32)

    
    # plotting the result graphs
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig("Accuracy plot.jpg")

    
    # plotting the result graphs
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("Loss plot.jpg")

    
    # function for running different statements in the model
    def predict_sentiment(text):
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw,maxlen=200)
        prediction = int(model.predict(tw).round().item())
        return factorized_sentiment[1][prediction]

    
    # importing complete data set and converting it to lower case
    # df_test = pd.read_csv("./complete_data.csv")
    df_test = pd.read_csv("./complete_data.csv")
    # df_test['Review'] = df_test['Review'].str.lower()
    # df_test['Review'] = df_test['Review'].to_string(na_rep='').lower()
    df_test["Review"]= df_test["Review"].map(str)
    df_test["Review"] = df_test["Review"].apply(str.lower)

    # Running data set on the new model
    count = 0
    df_test["Sentiment"] = np.nan
    for i in df_test["Review"]:
        df_test["Sentiment"][count] = predict_sentiment(i)
        count += 1

    # Converting data frame to a csv file
    df_test.to_csv('final_data.csv', index=False)
    
    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    textt = " ".join(review for review in df_test.Review)
    wordcloud = WordCloud(stopwords=stopwords).generate(textt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png')
    

def main():
    os.system("cls")
    print ("\nRunning Sentiment Analysis For:\n", scraper_list)
    combined_scrappers()

    # print ("\nPredicting Sentiment Using ML Model\n")
    # ml_model()

    print("\n\nFinal Data CSV Created!!")

if __name__ == "__main__":
    main()