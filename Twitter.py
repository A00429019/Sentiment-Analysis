from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import train as s

ckey = "EK0z3CHiUl3yZ0wLA7XgR224m"
csecret = "PZ32e9Z9yDEsclbv7nHAgFftBXk28UTwUEGPyyEccWpfVlZRMV"
atoken = "2900374860-62sDTdZiP9olajZAceDY4lMyeMU6NHC1RuionWw"
asecret = "VQzwSYOdogv1fJeceiKYNkDnsooZUl4U5AQePOa2rLEf0"


class StdOutListener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            fw = open("tweets.txt", "a")
            fw.write(sentiment_value + "\n")
            fw.close()
        return True

    def on_error(self, status):
        print(status)

while True:
    try:
        l = StdOutListener()
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        stream = Stream(auth, l)
        stream.filter(track=["love"])
    except:
        continue