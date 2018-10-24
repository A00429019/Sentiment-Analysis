from nltk.tokenize import word_tokenize
import nltk, random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifier = classifiers

    def classify(self, featureset):
        votes = []
        for c in self._classifier:
            v = c.classify(featureset)
            votes.append(v)
        return mode(votes)

    def confidence(self, featureset):
        votes = []
        for c in self._classifier:
            v = c.classify(featureset)
            votes.append(v)
        choice_vote = votes.count(mode(votes))
        conf = choice_vote/len(votes)
        return conf

neg_words = open(r"C:\Users\Acer\Desktop\PycharmProjects\YouTube\Sentiment\negative.txt", "r").read()
pos_words = open(r"C:\Users\Acer\Desktop\PycharmProjects\YouTube\Sentiment\positive.txt", "r").read()

documents = []
# all_words = []
#
for w in pos_words.split("\n"):
    documents.append((w, "pos"))
#     wordz = word_tokenize(w)
#     for x in wordz:
#         all_words.append(x.lower())
#
for w in neg_words.split("\n"):
    documents.append((w, "neg"))
#     wordz = word_tokenize(w)
#     for x in wordz:
#         all_words.append(x.lower())
#
random.shuffle(documents)
# random.shuffle(all_words)

# all_words = nltk.FreqDist(all_words)
# word_features = list(all_words.keys())[:5000]

word_features_f = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for x in word_features:
        features[x] = (x in words)
    return features

featuresets = [(find_features(rev), category) for rev, category in documents]

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# NBC = nltk.NaiveBayesClassifier.train(training_set)
# print("Naive Bayes:", (nltk.classify.accuracy(NBC, testing_set))*100)
# NBC.show_most_informative_features(15)

open_file = open("NaiveBayes.pickle", "rb")
NBC = pickle.load(open_file)
open_file.close()

# MNB = SklearnClassifier(MultinomialNB())
# MNB.train(training_set)
# print("Multinomial NB:", (nltk.classify.accuracy(MNB, testing_set))*100)

open_file = open("MNB.pickle", "rb")
MNB = pickle.load(open_file)
open_file.close()

# BNB = SklearnClassifier(BernoulliNB())
# BNB.train(training_set)
# print("Bernoulli NB:", (nltk.classify.accuracy(BNB, testing_set))*100)

open_file = open("BNB.pickle", "rb")
BNB = pickle.load(open_file)
open_file.close()

# LR = SklearnClassifier(LogisticRegression())
# LR.train(training_set)
# print("Logistic Regression:", (nltk.classify.accuracy(LR, testing_set))*100)

open_file = open("LR.pickle", "rb")
LR = pickle.load(open_file)
open_file.close()

# Support_vector = SklearnClassifier(SVC())
# Support_vector.train(training_set)
# print("Support Vector:", (nltk.classify.accuracy(Support_vector, testing_set))*100)

open_file = open("Support_vector.pickle", "rb")
Support_vector = pickle.load(open_file)
open_file.close()

# Linear = SklearnClassifier(LinearSVC())
# Linear.train(training_set)
# print("Linear SVC:", (nltk.classify.accuracy(Linear, testing_set))*100)

open_file = open("Linear.pickle", "rb")
Linear = pickle.load(open_file)
open_file.close()

# Nu = SklearnClassifier(NuSVC())
# Nu.train(training_set)
# print("NU SVC:", (nltk.classify.accuracy(Nu, testing_set))*100)

open_file = open("Nu.pickle", "rb")
Nu = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(NBC, MNB, BNB, LR, Support_vector, Linear, Nu)
# print("Voted Classifier:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)