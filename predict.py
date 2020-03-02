import tweepy
import pandas as pd
from pathlib import Path
from fastai.text import *
from fastai.tabular import *

path = Path('./')

# Fill in before running
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

def parse_tweet(raw_tweet):
    parsed_tweet = {}

    parsed_tweet["id"] = raw_tweet["id"]

    # Style
    parsed_tweet["text"] = raw_tweet["text"]
    parsed_tweet["num_mentions"] = len(raw_tweet["entities"]["user_mentions"])
    parsed_tweet["num_hashtags"] = len(raw_tweet["entities"]["hashtags"])
    parsed_tweet["num_urls"] = len(raw_tweet["entities"]["urls"])

    if "media" in raw_tweet["entities"].keys():
        parsed_tweet["has_media"] = True
    else:
        parsed_tweet["has_media"] = False

    # Propagation
    parsed_tweet["num_likes"] = raw_tweet["favorite_count"]
    parsed_tweet["num_retweets"] = raw_tweet["retweet_count"]

    # Credibility
    parsed_tweet["user_verified"] = raw_tweet["user"]["verified"]
    parsed_tweet["user_no_profile_image"] = raw_tweet["user"]["default_profile_image"]
    parsed_tweet["user_num_friends"] = raw_tweet["user"]["friends_count"]
    parsed_tweet["user_num_followers"] = raw_tweet["user"]["followers_count"]
    parsed_tweet["user_num_lists"] = raw_tweet["user"]["listed_count"]
    parsed_tweet["user_num_tweets"] = raw_tweet["user"]["statuses_count"]
    parsed_tweet["user_num_friends"] = raw_tweet["user"]["friends_count"]
    parsed_tweet["user_num_favourite_tweets"] = raw_tweet["user"]["favourites_count"]
    parsed_tweet["user_protected"] = raw_tweet["user"]["protected"]

    if raw_tweet["coordinates"] != None:
        parsed_tweet["has_location"] = True
    else:
        parsed_tweet["has_location"] = False

    # For filtering
    parsed_tweet["language"] = raw_tweet["lang"]

    parsed_tweet["annotation"] = None
    return parsed_tweet

tweet_id = input("Enter tweet ID: ")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitter_api = tweepy.API(auth)
tweet = twitter_api.get_status(tweet_id)

learn = load_learner(path, file = 'model_clas.pkl')
print(learn.predict(tweet.text))

parsed_tweet = parse_tweet(tweet._json)
tweets_df = pd.DataFrame.from_dict([parsed_tweet])
learn_tab = load_learner(path, file = 'model_clas_tab.pkl')
print(learn_tab.predict(tweets_df.iloc[0]))