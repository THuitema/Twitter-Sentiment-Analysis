# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
import numpy as np
import nlp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# dataset: https://github.com/huggingface/datasets/tree/master/datasets/emotion
# {'love', 'surprise', 'fear', 'joy', 'anger', 'sadness'}

dataset = nlp.load_dataset('emotion') # dictionary: train, validation, test
train_data = dataset['train'] # dict: text, label
validation_data = dataset['validation'] # dict: text, label
test_data = dataset['test'] # dict: text, label

def split_data(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels

(train_x, train_y) = split_data(train_data)
print(train_x[0])
print(train_x[1])

stemmer = PorterStemmer()
def prep_data(data):
    tokenized = [word_tokenize(tweet) for tweet in data] # tokenize: split sentences into words
    stopwords_removed = [[word for word in tweet if not word in stopwords.words('english')] for tweet in tokenized] # remove words with no significance
    stemmed = [[stemmer.stem(word) for word in tweet] for tweet in stopwords_removed] # shorten words to their stem
    data = [' '.join(tweet) for tweet in stemmed] # reform sentences
    print(data[0])
    print(data[1])
    tokenizer = Tokenizer(10000, oov_token='<UNK>')  # filter param default removes stopwords; oov_token replaces words not in 10000 most common
    tokenizer.fit_on_texts(data)  # assigns number to each word (updates tokenizer, not train_x)
    data = tokenizer.texts_to_sequences(data)  # converts text to integer values
    padded = pad_sequences(data, maxlen=50, padding='post', truncating='post')  # makes each tweet 50 words

    return padded


train_x = prep_data(train_x[:10])
print(train_x[0])
print(train_x[1])



# tokenizer = Tokenizer(10000, oov_token='<UNK>') # filter param default removes stopwords; oov_token replaces words not in 10000 most common
# tokenizer.fit_on_texts(train_x) # assigns number to each word (updates tokenizer, not train_x)
# train_x = tokenizer.texts_to_sequences(train_x) # converts text to integer values
# train_x = pad_sequences(train_x, maxlen=50, padding='post', truncating='post') # makes each tweet 50 words
#
# print(train_x[0])
# print(train_x[1])


# TODO: save train, test, and validation data as pickle files

