#!/usr/bin/env python3

"""
Face Finder for Face Recognition

This script contains the ModelTrainer class which is used to train and evaluate
different machine learning models for face recognition.

"""

from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from data_processor import DataProcessor
from report_manager import ReportManager


class ModelTrainer(DataProcessor, ReportManager):
    def __init__(self):
        DataProcessor.__init__(self)
        ReportManager.__init__(self)

    def naives_bayes(self, training_features: List[float], training_labels: List[int],
                     testing_features: List[float], testing_labels: List[int]
                     ) -> None:
        """
        Trains and evaluates a Gaussian Naive Bayes classifier.

        Args:
            training_features (List): Training feature vectors.
            training_labels (List): Training labels.
            testing_features (List): Testing feature vectors.
            testing_labels (List): Testing labels.
        """

        gnd = GaussianNB()
        gnd.fit(training_features, training_labels)  # Training the model
        predictions = gnd.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'Naives Bayes'

    def ann(self, training_features: List[float], training_labels: List[int],
            testing_features: List[float], testing_labels: List[int]
            ) -> None:

        classifier = MLPClassifier(hidden_layer_sizes=(
            100,), max_iter=10000, activation='relu', solver='adam', random_state=1)
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)

        return report, testing_labels,  predictions, 'ANN'

    def knn(self, training_features: List[float], training_labels: List[int],
            testing_features: List[float], testing_labels: List[int]
            ) -> tuple[List[int], List[int], str]:
        """
        Trains and evaluates a k-Nearest Neighbors (kNN) classifier.

        Args:
            training_features (List): Training feature vectors.
            training_labels (List): Training labels.
            testing_features (List): Testing feature vectors.
            testing_labels (List): Testing labels.
        """

        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(training_features, training_labels)
        predictions = knn.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)

        return report, testing_labels, predictions, 'KNN'

    def support_vector_machine(self, training_features, training_labels, testing_features, testing_labels):
        classifier = SVC(kernel='rbf', C=1)
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'SVM'

    def decision_tree(self, training_features, training_labels, testing_features, testing_labels):
        tree_classifier = DecisionTreeClassifier(random_state=0)
        tree_classifier.fit(training_features, training_labels)
        predictions = tree_classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'Decision Tree'
