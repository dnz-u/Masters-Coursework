import os
import numpy as np
from PIL import Image


import pandas as pd

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

import argparse

import csv


dataset_root = "./DATASETS/model1_dataset"
RESULT_FILE_NAME = "even_data_res"


def load_data(path: str):

    np_file = np.load(path)

    arr1 = np_file['X_train']
    arr2 = np_file['y_train']
    arr3 = np_file['X_val']
    arr4 = np_file['y_val']
    arr5 = np_file['X_test']
    arr6 = np_file['y_test']
    
    return arr1, arr2, arr3, arr4, arr5, arr6


def accs(y_pred, y_true, verbose=None):
    labels = {0: 0 , 1: 0, 2: 0}
    true_predictions = {0: 0 , 1: 0, 2: 0}
    
    for pred, truth in zip(y_pred, y_true):
        labels[truth] += 1
        if pred == truth:
            true_predictions[pred] += 1
    
    overall_acc = sum(true_predictions.values()) / sum(labels.values())

    if verbose:
        print(f"0-Covid:{true_predictions[0]}/{labels[0]}, 1-Normal:{true_predictions[0]}/{labels[0]}, 2-Pneumonia:{true_predictions[0]}/{labels[0]}")
        print(f"Accuracy: {overall_acc}")
        
    return labels, true_predictions, overall_acc


def append_string_to_csv(file_name, string):
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([string])
    
    if not file_exists:
        with open(file_name, 'r+') as csv_file:
            content = csv_file.read()
            csv_file.seek(0, 0)
            
            if not content:
                # Write the header row if the file is empty
                writer = csv.writer(csv_file)
                writer.writerow([
                    'Column Name1',
                    'Column Name2',])



def run(f_path, classifier_type):
    
    classifiers = {
        'svm_classifier': SVC(random_state=42),

        'dt_classifier': DecisionTreeClassifier(random_state=42),

        'gmm_classifier': GaussianMixture(random_state=42, n_components=3),

        'knn_classifier': KNeighborsClassifier(),

        'adaboost_classifier': AdaBoostClassifier(n_estimators=100, random_state=42),

        'bagging_svm_classifier': BaggingClassifier(
                                            base_estimator=SVC(random_state=42),
                                            n_estimators=10, random_state=42),

        'bagging_dt_classifier': BaggingClassifier(
                                            base_estimator=DecisionTreeClassifier(random_state=42),
                                            n_estimators=10, random_state=42),

        'random_forest_classifier': RandomForestClassifier(n_estimators=100, random_state=42),

        'bagging_gmm_classifier': BaggingClassifier(
                                            base_estimator=GaussianMixture(random_state=42, n_components=3),
                                            n_estimators=10, random_state=42),

        'bagging_knn_classifier': BaggingClassifier(
                                            base_estimator=KNeighborsClassifier(),
                                            n_estimators=10, random_state=42),

        'bagging_adaboost_classifier': BaggingClassifier(
                                            base_estimator=AdaBoostClassifier(n_estimators=100, random_state=42),
                                            n_estimators=10,
                                            random_state=42)
    }
    
    # read arrays
    arrays_path = dataset_root + '/' + f_path
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(arrays_path)
    
    # classifier_type: str, script argument
    clf = classifiers[classifier_type]
    
    # train the model
    clf.fit(X_train, y_train)
    
    # calculate accuraciest
    train_predictions = clf.predict(X_train)
    val_predictions = clf.predict(X_val)
    test_predictions = clf.predict(X_test)
    
    _, _, train_acc = accs(train_predictions, y_train, verbose=None)
    _, _, val_acc = accs(val_predictions, y_val, verbose=None)
    test_labels, test_true_preds, test_acc = accs(test_predictions, y_test, verbose=None)
    
    data_no = f_path.split("-")[2]
    
    csv_list = [
        classifier_type,
        data_no,
        str(train_acc), str(val_acc), str(test_acc), 
        str(test_true_preds[0]), str(test_labels[0]),
        str(test_true_preds[1]), str(test_labels[1]),
        str(test_true_preds[2]), str(test_labels[2]),
    ]    
    
    csv_str = ",".join(csv_list)
    append_string_to_csv(RESULT_FILE_NAME, csv_str)
    

def parser():
    parser = argparse.ArgumentParser(description='CS550 Final Project')

    parser.add_argument('-f', '--file', type=str, help='File path')
    parser.add_argument('-c', '--classifier', type=str, help='Classifier type')

    args = parser.parse_args()

    file_path = args.file
    classifier_type = args.classifier
    
    return file_path, classifier_type

    
if __name__ == '__main__':    
    f_path, clf_type = parser()
    run(f_path, clf_type)
