import nltk.tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from langdetect import detect
from googletrans import Translator
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

s_time = time.time()
train_data = pd.read_csv("assignment-comp3222-comp6246-mediaeval2015-dataset/mediaeval-2015-trainingset.txt",
                         sep="	")
test_data = pd.read_csv("assignment-comp3222-comp6246-mediaeval2015-dataset/mediaeval-2015-testset.txt", sep="	")

# train_data.to_csv("train_data.csv", index=None)
# test_data.to_csv("test_data.csv", index=None)

# print(train_data.info())
# print(test_data.info())
# Get the data frame for both training and testing data
train_df = pd.DataFrame(data=train_data)
test_df = pd.DataFrame(data=test_data)

# visualize the training data based on the event
train_df.rename(columns={"imageId(s)": "images"}, inplace=True)
# train_df["images"] = train_df["images"].str.split("_").str[0]
# event_count_train = train_df["images"].value_counts().reset_index().rename(
#     columns={"index": "event", "images": "number"})
# event_count_train.plot.barh(x="event", y="number", alpha=0.5, title="Event count for training data")
# plt.show()
# print(event_count_train)

# visualize the testing data based on the event
test_df.rename(columns={"imageId(s)": "images"}, inplace=True)
# test_df["images"] = test_df["images"].str.split("_").str[0] event_count_test = test_df["images"].value_counts(
# ).reset_index().rename(columns={"index": "event", "images": "number"}) event_count_test.plot.barh(x="event",
# y="number", alpha=0.5, title="Event count for testing data")


# plt.show()
# print(event_count_test)

# Preprocess the training data
# Change the label of humor to fake and map labels to numerical data. real posts as 1, fake as 0
train_df["label"] = train_df["label"].map({'real': 1, 'fake': 0, 'humor': 0})
test_df["label"] = test_data["label"].map({'real': 1, 'fake': 0, 'humor': 0})


# Remove emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Remove URLs
def remove_URLs(string):
    result1 = re.sub(r'http\S+', "", string)
    result = re.sub(r'\\\/\S+', "", result1)
    return result


# remove special characters
def remove_special_char(string):
    result = re.sub(r'&|\\n', "", string)
    return result


# Translate all tweetTexts to English
def translate_to_english(string):
    if string is not None:
        try:
            tr = Translator()
            string = tr.translate(string)
        except AttributeError:
            print("can not translate")

    return string


def remove_whitespace(string):
    return " ".join(string.split())


def remove_at_username(string):
    string = re.sub(r'@\w*', "", string)
    return string


def lemmatising(string):
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(x) for x in tokenizer.tokenize(string)])


def preprocess(string):
    string = string.lower()
    string = remove_emoji(string)
    string = remove_URLs(string)
    string = remove_at_username(string)
    string = remove_special_char(string)
    string = remove_whitespace(string)
    return string


train_df["cleanedText"] = train_df["tweetText"].apply(preprocess)
test_df["cleanedText"] = test_df["tweetText"].apply(preprocess)

# remove duplicate posts and remove empty text rows
train_df.drop_duplicates(subset=["cleanedText"], keep="first", inplace=True, ignore_index=False)
train_df["cleanedText"].replace("", np.nan, inplace=True)
train_df.dropna(subset=["cleanedText"], inplace=True)
test_df.drop_duplicates(subset=["cleanedText"], keep="first", inplace=True, ignore_index=False)
test_df["cleanedText"].replace("", np.nan, inplace=True)
test_df.dropna(subset=["cleanedText"], inplace=True)

# remove stopwords
stopwords = nltk.corpus.stopwords.words()
stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...', '!', ',', '/', ])
def remove_stopwords(string):
    word_tokens = word_tokenize(string)
    filtered_sent = [w for w in word_tokens if not w.lower() in stopwords]
    result = " ".join(filtered_sent)
    return result


def count_hashtags(string):
    count = 0
    for hash_tag in re.findall('#(\w+)', string):
        count += 1
    return str(count)


train_df["num_hashtags"] = train_df["cleanedText"].apply(count_hashtags)
train_df["cleanedText"] = train_df["cleanedText"].apply(remove_stopwords)
train_df["cleanedText"] = train_df["cleanedText"].apply(lemmatising)
train_df["cleanedText"] = train_df["cleanedText"].apply(remove_whitespace)
train_df["textLength"] = train_df["cleanedText"].str.len()
train_df["textLength"] = [str(length) for length in train_df["textLength"]]
train_df["language"] = train_df["cleanedText"].apply(lambda text: detect(text))

test_df["num_hashtags"] = test_df["cleanedText"].apply(count_hashtags)
test_df["cleanedText"] = test_df["cleanedText"].apply(remove_stopwords)
test_df["cleanedText"] = test_df["cleanedText"].apply(lemmatising)
test_df["cleanedText"] = test_df["cleanedText"].apply(remove_whitespace)
test_df["textLength"] = test_df["cleanedText"].str.len()
test_df["textLength"] = [str(length) for length in test_df["textLength"]]
test_df["language"] = test_df["cleanedText"].apply(lambda text: detect(text))
train_df.to_csv("train_data1.csv", index=None)
test_df.to_csv("test_data1.csv", index=None)

labels_train = train_df["label"]
features_train = train_df["cleanedText"] + " " + \
                 train_df["num_hashtags"] + " " + \
                 train_df["language"] + " " + train_df["username"] + " " + train_df["timestamp"]
labels_test = test_df["label"]
features_test = test_df["cleanedText"] + " " + \
                test_df["num_hashtags"] + " " + \
                test_df["language"] + " " + test_df["username"] + " " + test_df["timestamp"]


# Function to calculate F1 score
def calculate_F1(labels_test, pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for truth, pred in zip(labels_test, pred):
        if truth == 0 and pred == 0:
            tp += 1
        if truth == 1 and pred == 0:
            fp += 1
        if truth == 1 and pred == 1:
            tn += 1
        if truth == 0 and pred == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# TF-IDF + RF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_train = tfidf_vectorizer.fit_transform(features_train)
tfidf_test = tfidf_vectorizer.transform(features_test)
rf_clf = RandomForestClassifier()
rf_clf.fit(tfidf_train, labels_train)
rf_pred = rf_clf.predict(tfidf_test)
print("TF-IDF + RF f1:", calculate_F1(labels_test, rf_pred))

# TF-IDF + MNB
mnb_clf = MultinomialNB()
mnb_clf.fit(tfidf_train, labels_train)
cnb_pred = mnb_clf.predict(tfidf_test)
print("TF-IDF + MNB f1:", calculate_F1(labels_test, cnb_pred))

# TF-IDF + SGD
sgd_clf = SGDClassifier()
sgd_clf.fit(tfidf_train, labels_train)
sgd_pred = sgd_clf.predict(tfidf_test)
print("TF-IDF + SGD f1:", calculate_F1(labels_test, sgd_pred))

# n-grams + RF
n_gram_vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2))
n_gram_train = n_gram_vectorizer.fit_transform(features_train)
n_gram_test = n_gram_vectorizer.transform(features_test)
rf_clf.fit(n_gram_train, labels_train)
rf_pred = rf_clf.predict(n_gram_test)
print("n-grams + RF f1:", calculate_F1(labels_test, rf_pred))

# n-grams + MNB
mnb_clf.fit(n_gram_train, labels_train)
cnb_pred = mnb_clf.predict(n_gram_test)
print("n-grams + MNB f1:", calculate_F1(labels_test, cnb_pred))

# n-grams + SGB
sgd_clf.fit(n_gram_train, labels_train)
sgd_pred = sgd_clf.predict(n_gram_test)
print("n-grams + SGB f1:", calculate_F1(labels_test, sgd_pred))
e_time = time.time()
print("Processing: " + str(e_time - s_time) + 's')
