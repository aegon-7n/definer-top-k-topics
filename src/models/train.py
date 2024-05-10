# import logging
import os
import sys

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

import mlflow
from mlflow.tracking import MlflowClient

preprocessing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
sys.path.append(preprocessing_dir)

import get_comments as gc
import preprocessing_text as pt
import cluster_train as cl

dir = str(os.getcwd())
config_path = f'{dir}/config/params_all.yaml'
config = yaml.safe_load(open(config_path))['train']
SEED = config['SEED']

# logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.DEBUG)

def get_latest_version_number(config_name, client):
    model_versions = client.search_model_versions(f"name='{config_name}'")
    return max(model_version.version for model_version in model_versions)

def main():
    comments = gc.get_all_comments(**config['comments'])
    language = config['stopwords']
    stop_words = set(stopwords.words(language))
    cleaned_comments = pt.get_clean_text(comments, stop_words)

    tfidf = TfidfVectorizer(**config['tf_model']).fit(cleaned_comments)
    mtx = pt.vectorize_text(cleaned_comments, tfidf)
    cluster_labels = cl.get_clusters(mtx, random_state=SEED, **config['clustering'])

    X_train, X_test, y_train, y_test = train_test_split(mtx, cluster_labels, **config['cross_val'], random_state=SEED)
    clf_lr = LogisticRegression(**config['model'])

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(config['name_experiment'])
    with mlflow.start_run():
        clf_lr.fit(X_train, y_train)

        mlflow.log_param('f1',
                         cl.get_f1_score(y_test, clf_lr.predict(X_test), set(cluster_labels)))
        mlflow.log_param('accuracy',
                         accuracy_score(y_test, clf_lr.predict(X_test)))
        mlflow.log_param('precision',
                         cl.get_precision_score(y_test, clf_lr.predict(X_test), set(cluster_labels)))
        mlflow.sklearn.log_model(tfidf,
                                 artifact_path="vector",
                                 registered_model_name=f"{config['model_vec']}")
        mlflow.sklearn.log_model(clf_lr,
                                 artifact_path='model_lr',
                                 registered_model_name=f"{config['model_lr']}")
        mlflow.log_artifact(local_path='./src/models/train.py',
                            artifact_path='code')
        mlflow.end_run()

    with open(config_path, 'r') as file:
        yaml_file = yaml.safe_load(file)

    client = MlflowClient()
    last_version_lr = get_latest_version_number(config['model_lr'], client)
    last_version_vec = get_latest_version_number(config['model_vec'], client)

    yaml_file['predict']["version_lr"] = int(last_version_lr)
    yaml_file['predict']["version_vec"] = int(last_version_vec)

    with open(config_path, 'w') as file:
        yaml.dump(yaml_file, file, encoding='UTF-8', allow_unicode=True)

if __name__ == "__main__":
    main()