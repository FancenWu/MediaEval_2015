import pandas as pd
import matplotlib.pyplot as plt
import time
import nltk
import re
from langdetect import detect
from googletrans import Translator

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


def preprocess(string):
    string = remove_emoji(string)
    string = remove_URLs(string)
    # string = translate_to_english(string)
    string = remove_special_char(string)
    return string


train_df["cleanedText"] = train_df["tweetText"].apply(preprocess)
test_df["cleanedText"] = test_df["tweetText"].apply(preprocess)

# remove duplicate posts
train_df.drop_duplicates(subset=["cleanedText"], keep="first", inplace=True, ignore_index=False)

# remove stopwords
stopwords = nltk.corpus.stopwords.words()
stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...'])
def remove_stopwords(string):
    result = ' '.join([w for w in string.split() if w not in stopwords])
    return result


train_df["cleanedText"] = train_df["cleanedText"].apply(remove_stopwords)
train_df["length"] = train_df["cleanedText"].str.len()
# train_df.to_csv("train_data1.csv", index=None)
print(train_df["length"].min())
e_time = time.time()
print("Processing: " + str(e_time - s_time) + 's')
