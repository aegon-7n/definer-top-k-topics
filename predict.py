import logging
import os

import numpy as np
import pandas as pd
import yaml

import mlflow
import src.get_comments as gc
import src.preprocessing_text as pt

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


dir = '/home/aegon/Documents/Development/Projects/topic-sentiment-explorer/'
config_path = os.path.join(dir + 'config/params_all.yaml')
config = yaml.safe_load(open(config_path))['train']
os.chdir(config['dir_folder'])
SEED = config['SEED']

logging.basicConfig(filename='log/app.log', 
                    filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def save_topics(data, y_predict, vector_model, num_words, name_file):
    topics = {}
    for i in list(set(y_predict)):
        ind_ = data[np.where(y_predict == i)[0]].sum(axis=0).argsort()[-num_words:]
        topics[i] = [vector_model.get_feature_names()[i] for i in ind_]
    pd.DataFrame().from_dict(topics).to_csv(name_file, encoding='cp1251')


def main():
    comments = gc.get_all_comments(**config['comments'])

    mlflow.set_tracking_uri("http://localhost:5000")
    model_uri_lr = f"models:/{config['model_lr']}/{config['version_lr']}"
    model_uri_tf = f"models:/{config['model_vec']}/{config['version_vec']}"

    model_lr = mlflow.sklearn.load_model(model_uri_lr)
    tfidf = mlflow.sklearn.load_model(model_uri_tf)

    language = config['stopwords']
    stop_words = set(stopwords.words(language))
    comments_clean = pt.get_clean_text(comments, stop_words)

    X_matrix = pt.vectorize_text(comments_clean, tfidf)
    
    save_topics(X_matrix, model_lr.predict(X_matrix), tfidf, config['num_top'], config['name_file'])


if __name__ == "__main__":
    main()