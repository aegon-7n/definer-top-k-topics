import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tokenize import word_tokenize


def remove_emoji(sentence):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', sentence)


def remove_links(sentence):
    link_pattern = r'(http\S+|bit\.ly/\S+|www\S+)'
    sentence = re.sub(link_pattern, '', sentence)
    sentence = sentence.strip('[link]')
    return sentence


def preprocessing(sentence, stop_words, lemmatizer):
    sentence = remove_emoji(sentence)
    sentence = remove_links(sentence)

    str_pattern = re.compile("\r\n")
    sentence = str_pattern.sub(r'', sentence)

    sentence = re.sub('(((?![а-яА-Я ]).)+)', ' ', sentence)

    sentence = [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word not in stop_words]
    sentence = ' '.join(sentence)
    return sentence


def get_clean_text(data, stop_words):
    lemmatizer = WordNetLemmatizer()
    comments = [preprocessing(sentence, stop_words, lemmatizer) for sentence in data]
    comments = [comm for comm in comments if len(comm) > 2]
    return comments


def vectorize_text(data, tfidf):
    mtx = tfidf.transform(data).toarray()
    mask = (np.nan_to_num(mtx) != 0).any(axis=1)
    return mtx[mask]