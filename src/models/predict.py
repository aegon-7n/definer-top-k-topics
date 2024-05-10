# import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

import mlflow

preprocessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
sys.path.append(preprocessing_dir)

import get_comments as gc
import preprocessing_text as pt
import cluster_train as cl

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


dir = str(os.getcwd())
config_path = f'{dir}/config/params_all.yaml'
config = yaml.safe_load(open(config_path))['predict']
SEED = config['SEED']

# logging.basicConfig(filename='log/app.log', 
#                     filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.DEBUG)

def save_topics(data, y_predict, vector_model, num_words, name_file):
    current_directory = os.getcwd()
    name_file = os.path.abspath(os.path.join(current_directory, name_file))

    topics = {}
    for i in list(set(y_predict)):
        ind_ = data[np.where(y_predict == i)[0]].sum(axis=0).argsort()[-num_words:]
        topics[i] = [vector_model.get_feature_names_out()[i] for i in ind_]
    pd.DataFrame().from_dict(topics).to_csv(name_file, index=False)

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