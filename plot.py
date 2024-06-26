import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class Plot:
    """
    This class contains methods to plot evaluation metrics for a multiclass classification problem.
    The methods include (evaluation_metrics) plotting a confusion matrix 
    and (roc_curve) plotting ROC curve for a given model.

    """

    def __init__(self, training_features=[], training_labels=[],
                 testing_features=[], testing_labels=[]):
        self.training_features = training_features
        self.training_labels = training_labels
        self.testing_features = testing_features
        self.testing_labels = testing_labels

    def evaluation_metrics(self, testing_labels, predictions,
                           model_name, output_dir='confusion_matrices_plt') -> tuple[float, np.ndarray, str]:
        """
        Plots a confusion matrix for a multiclass classification problem.

        Args:
            testing_labels (List): True labels of the testing set.
            predictions (List): Predicted labels for the testing set.
            model_name (str): Name of the model for which the confusion matrix is being plotted.
            output_dir (str): Directory to save the confusion matrix plot. Default is 'confusion_matrices'.

        Returns:
            tuple: A tuple containing the accuracy score, confusion matrix, and model name.
        """

        accuracy = accuracy_score(
            testing_labels, predictions)
        conf_matrix = confusion_matrix(testing_labels, predictions)

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        file_path = os.path.join(output_dir, f'CM_{model_name}.png')
        plt.savefig(file_path)
        plt.pause(.1)
        plt.close()

        return accuracy, conf_matrix, model_name

    def plt_roc_curve(self, testing_labels, predictions,
                      model_name, output_dir='roc_curves_plt'
                      ) -> tuple[dict, dict, dict]:
        """
        Plots One-vs-Rest (OvR) ROC curves for a multiclass classification problem.

        Args:
            testing_labels (List): True labels of the testing set.
            predictions (List): Predicted probabilities or decision function scores for the testing set.
            model_name (str): Name of the model for which the ROC curves are being plotted.
            output_dir (str): Directory to save the ROC curve plot. Default is 'roc_curves'.

        Returns:
            tuple: A tuple containing the false positive rate, true positive rate, and ROC area.
        """

        lb = LabelBinarizer()
        lb.fit(testing_labels)
        y_true = lb.transform(testing_labels)

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(lb.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predictions[:, 0])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curves for each class
        plt.figure()
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve (class {lb.classes_[i]}) (area = {roc_auc[i]:0.2f})')

        os.makedirs(output_dir, exist_ok=True)
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        file_path = os.path.join(
            output_dir, f'ROC_{model_name}.png')
        plt.savefig(file_path)
        plt.close()
        return fpr, tpr, roc_auc

    def plt_avg_roc(self, classifiers, X_test, y_test, output_dir='avg_roc_curves_plt'):
        """
        Plots the average ROC curve for the given classifiers.

        Parameters:
        classifiers (list): List of trained classifiers.
        X_test (array-like): Test features.
        y_test (array-like): True labels for the test data.
        """
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, (clf, color) in enumerate(zip(classifiers, colors)):
            model_name = type(clf).__name__
            probas_ = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(
                y_test, probas_[:, 1], pos_label=1)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.7, color=color,
                     label=f'{model_name} ROC fold {i+1} (AUC = {roc_auc:.2f})')

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        os.makedirs(output_dir, exist_ok=True)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        file_path = os.path.join(output_dir, 'avg_roc_curve.png')
        plt.savefig(file_path)
        plt.close()

    def plot_classifier(self):
        classifiers = [
            KNeighborsClassifier(),
            SVC(probability=True),
            DecisionTreeClassifier(),
            GaussianNB(),
            MLPClassifier(max_iter=10000)
        ]

        for clf in classifiers:
            clf.fit(self.training_features, self.training_labels)

        self.plt_avg_roc(
            classifiers, self.testing_features, self.testing_labels)
