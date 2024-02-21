import mlflow

import numpy as np
import argparse
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# import joblib
from azureml.core import Dataset, Run
import subprocess
from mlflow.models.signature import infer_signature
from azureml.core import Run
package_name4='sentence-transformers'

subprocess.check_call(['pip', 'install', package_name4])

from sentence_transformers import SentenceTransformer
import mlflow
from azureml.core import Workspace
from azureml.core import Workspace, Experiment


# Configure experiment
run = Run.get_context()
ws = run.experiment.workspace
##############setting mlflow tracking
import mlflow
tracking_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(tracking_uri)

from urllib.parse import urlparse

urlparse(mlflow.get_tracking_uri()).scheme


mlflow.create_experiment('mlflow_spammodel_pipeline')
mlflow.set_experiment('mlflow_spammodel_pipeline')


#######################################
# ws = Workspace(subscription_id='738974c0-9684-498b-bbd0-c53d6d04d964',resource_group='rg-oof-cus-d-001',workspace_name='ML-OOF-CUS-D')
# Setup Run
# ---------------------------------------

# Load the current run and ws
# tracking_uri = ws.get_mlflow_tracking_uri()
# mlflow.set_tracking_uri(tracking_uri)
# from urllib.parse import urlparse

# urlparse(mlflow.get_tracking_uri()).scheme


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--colsample-bytree', type=float)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--max-bin', type=int, default=0.15)

parser.add_argument('--max-depth', type=int, default=500)
# parser.add_argument('--min-split-gain', type=float, default=200)
parser.add_argument('--n-estimators', type=int, default=20)
parser.add_argument('--num-leaves', type=int, default=20)

parser.add_argument('--reg-alpha', type=float, default=0.7)
parser.add_argument('--reg-lambda', type=float, default=40)
parser.add_argument('--subsample', type=float, default=40)

args = parser.parse_args()


lgbm_params = {
    'colsample_bytree': args.colsample_bytree,

    'learning_rate': args.learning_rate,
    'max_bin': args.max_bin,
    'max_depth': args.max_depth,
    # 'min_split_gain': args.min_split_gain,
    'n_estimators': args.n_estimators,
    'num_leaves': args.num_leaves,
    'reg_alpha': args.reg_alpha,
    'reg_lambda': args.reg_lambda,
    'subsample': args.subsample
}



# Load data
# ---------------------------------------

# Get a dataset by id
dataset = Dataset.get_by_id(ws, id=args.data)

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()



# Preprocessing
# ---------------------------------------

   

print('Encoding of exsisitng data happening.....')

import os
#print(os.getcwd())
# os.chdir("../Bert_model")
# bert_model = SentenceTransformer('../Bert_model/all-mpnet-base-v2')
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
# bert_model = SentenceTransformer('../Bert_model/all-mpnet-base-v2')

     
#print(f'shape of dataset going for training: {df.shape[0]}')

body2 = list(df['preprocessedBody'])
tfidf_features = bert_model.encode(body2, show_progress_bar=True)

from mlflow.models.signature import infer_signature


######################################
# # Take a hold out set randomly
# X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['topic'], test_size=0.15, random_state=42,stratify=df['topic'])
# train_data = lgbm.Dataset(data=X_train, label=y_train)
# test_data  = lgbm.Dataset(data=X_test, label=y_test)

# full_train= lgbm.Dataset(data=tfidf_features, label=df['topic'])

# def train_model():
#     mlflow.lightgbm.autolog()
    

#     with mlflow.start_run() as my_run:
#         clf = lgbm.train(train_set=train_data,
#                  params=lgbm_params,
#                  valid_sets=[train_data, test_data], 
#                  valid_names=['train', 'val'],
                
               
           
#                  verbose_eval=20,
#                  )

            
        
         
#         preds = np.round(clf.predict(X_test))
#         signature = infer_signature(X_test, preds)
#         acc= ( accuracy_score(y_test, preds))
#         pr= (precision_score(y_test, preds))
#         rec=(recall_score(y_test, preds))
#         f1=(f1_score(y_test, preds))
#         mlflow.log_metric('accuracy', acc)
#         mlflow.log_metric('precison', pr)
#         mlflow.log_metric('recall', rec)
#         mlflow.log_metric('f1 score', f1)
#         mlflow.log_params(lgbm_params)
#         mlflow.lightgbm.log_model(clf, "spam-model", signature=signature)
#         #to log in pipeline
#         run = Run.get_context()
#         run.parent.log("Accuracy", acc)
#         run.parent.log("precison", pr)
#         run.parent.log("recall", rec)
#         run.parent.log("f1 score", f1)



#     print("MLflow run id: %s" % my_run.info.run_id)

#     # mlflow.register_model(my_run.info.run_id, "sk-learn-random-forest-reg")


#     # mlflow.lightgbm.log_model(
#     #     lgb_model=clf,
#     #     artifact_path="/",
#     #     signature=signature,
#     #     registered_model_name="mlflow-spammodel",
#     # )


#     return my_run

# mlflow.lightgbm.autolog()

# model_run = train_model()



######################################

#functions for training model for spam model
import lightgbm as lgbm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from azureml.core.model import Model


def log_metrics(clf, X_test, y_test, run):
    preds = np.round(clf.predict(X_test))
    run.log("accuracy (test)", accuracy_score(y_test, preds))
    run.log("precision (test)", precision_score(y_test, preds))
    run.log("recall (test)", recall_score(y_test, preds))
    run.log("f1 (test)", f1_score(y_test, preds))

def train_model():
    
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['topic'], test_size=0.15, random_state=42,stratify=df['topic'])
    train_data = lgbm.Dataset(data=X_train, label=y_train)
    test_data  = lgbm.Dataset(data=X_test, label=y_test)

    full_train= lgbm.Dataset(data=tfidf_features, label=df['topic'])



    mlflow.lightgbm.autolog()

    with mlflow.start_run() as my_run:

        evaluation_results = {}

        
        clf = lgbm.train(train_set=train_data,
                 params=lgbm_params,
                 valid_sets=[train_data, test_data], 
                 valid_names=['train', 'val'],
                 evals_result=evaluation_results,
                 num_boost_round=500
               
                )
        

        preds = np.round(clf.predict(X_test))
        acc= ( accuracy_score(y_test, preds))
        pr= (precision_score(y_test, preds))
        rec=(recall_score(y_test, preds))
        f1=(f1_score(y_test, preds))
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precison', pr)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1 score', f1)

    



    return my_run



model_run = train_model()




