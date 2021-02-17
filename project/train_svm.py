"""
Trains a support vector machine for land cover/ land usage.
The SVM is trained on the EuroSAT dataset and classifies image patches into
forest and non-forest.
"""

import config
import os
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from skimage import io


def get_train_test_split(df):
    """
    Divides the dataset into training and testing data, performing a 80:20 split.

    :param df:  data frame

    :return:    training data, testing data, training labels, testing labels
    """

    num_samples = df.shape[0]
    X = []
    y = []

    for i in range(0, num_samples):
        img_name = df.loc[i, 'Filename']
        img = io.imread(os.path.join(config.DATSET_PATH, str(img_name)))
        label = df.loc[i, 'Label']

        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((num_samples, -1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test


def load_dataframe():
    """
    Loads data frame from disk and preprocesses it for our use case, i.e. fusing
    non-forest classes together and balancing the dataset.

    :return:    preprocessed dataset.
    """
    # load data from csv file
    train_df = pd.read_csv(os.path.join(config.DATSET_PATH, "train.csv"))
    train_df.drop(columns=train_df.columns[0], inplace=True)

    # get class names
    with open(os.path.join(config.DATSET_PATH, "label_map.json"), "r") as file:
        class_names_encoded = json.load(file)
    class_names = list(class_names_encoded.keys())

    # stats: count number of class instances
    _, train_labels_count = np.unique(train_df['Label'], return_counts=True)

    # keep forest class, get the same amount of other classes
    df_forest = train_df.drop(np.where(train_df['Label'] != 1)[0], inplace=False)
    forest_count = train_labels_count[1]
    df_rest = train_df.drop(
        np.where(train_df['Label'] == 1)[0], inplace=False).iloc[0:forest_count]
    frames = [df_forest, df_rest]
    train_df_cleaned = pd.concat(frames, ignore_index=True)

    # getting the class distribution in the prepared dataset
    _, train_labels_count = np.unique(train_df_cleaned['Label'], return_counts=True)
    train_count_df = pd.DataFrame(data=train_labels_count)
    train_count_df['ClassName'] = class_names
    train_count_df.columns = ['Count', 'ClassName']
    train_count_df.set_index('ClassName', inplace=True)
    train_count_df.head()
    train_count_df.plot.bar()
    plt.title("Distribution of images per class")
    plt.ylabel("Count")
    plt.show()

    # set non forest class to alternative
    train_df_cleaned.loc[forest_count:(2 * forest_count), 'Label'] = 0
    return train_df_cleaned


def train_svm():
    """
    Trains a SVM on the preprocessed dataset, outputs its performance and stores
    the trained SVM
    """
    df = load_dataframe()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_pred)))

    # store trained SVM
    with open(config.SVM_PATH, 'wb') as file:
        pickle.dump(classifier, file)


if __name__ == "__main__":
    train_svm()
