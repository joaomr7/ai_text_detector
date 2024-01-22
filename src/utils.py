import os
import pickle
from pathlib import Path

from src.exception import CustomException
from src.paths import crate_directory_if_not_exist

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        crate_directory_if_not_exist(dir_path)

        with open(Path(file_path), 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e)
    
def load_object(file_path):
    try:
        with open(Path(file_path), 'rb') as file:
            return pickle.load(file)
    
    except Exception as e:
        raise CustomException(e)

def evaluate_model_performance(model, limiar, X, y):
    try:
        y_pred_score = model.predict_proba(X)[:, 1]
        y_pred = y_pred_score >= limiar

        auc_score = roc_auc_score(y, y_pred_score)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return (auc_score, accuracy, precision, recall, f1)

    except Exception as e:
        raise CustomException(e)
    
def format_model_metrics(auc, accuracy, precision, recall, f1):
    return f'auc: {auc:.2f}, accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1_score: {f1:.2f}'