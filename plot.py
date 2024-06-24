import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from itertools import chain


class Plot:
    """
    This class contains methods to plot evaluation metrics for a multiclass classification problem.
    The methods include (evaluation_metrics) plotting a confusion matrix 
    and (roc_curve) plotting ROC curve for a given model.

    """

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
        # os.makedirs(output_dir, exist_ok=True)

        # lb = LabelBinarizer()
        # lb.fit(testing_labels)
        # y_true = lb.transform(testing_labels)

        # # Ensure predictions is a list of arrays, reshape if necessary
        # for i, preds in enumerate(predictions):
        #     if preds.ndim == 1:
        #         predictions[i] = preds.reshape(-1, 1)
        # # print(predictions)
        # # print(lb.classes_)
        # # exit()

        # # Initialize dictionaries to store fpr, tpr for each class
        # fpr = {}
        # tpr = {}
        # roc_auc = {}

        # # Compute ROC curve and ROC area for each class
        # for i in range(len(lb.classes_)):
        #     mean_fpr = np.linspace(0, 1, 100)
        #     tpr[i] = np.zeros_like(mean_fpr)

        #     for preds in predictions:
        #         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], preds[:, i])
        #         tpr[i] += np.interp(mean_fpr, fpr[i], tpr[i])
        #         roc_auc[i] = auc(fpr[i], tpr[i])

        #     tpr[i] /= len(predictions)
        #     fpr[i] = mean_fpr
        # # print(fpr)
        # # print(tpr)
        # # exit()
        # # Plot the ROC curve for the average model
        # plt.figure(figsize=(8, 6))
        # colors = ['blue', 'red', 'green', 'purple', 'orange']
        # for i, color in zip(range(len(lb.classes_)), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
        #              label=f'ROC curve (class {lb.classes_[i]}) (area = {roc_auc[i]:0.2f})')

        # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(f'ROC Curves - {model_name}')
        # plt.legend(loc="lower right")
        # plt.grid(True)
        # plt.tight_layout()

        # # Save the plot
        # plt.savefig(os.path.join(output_dir, f'ROC_{model_name}.png'))
        # plt.close()

        # return fpr, tpr, roc_auc

        os.makedirs(output_dir, exist_ok=True)

        lb = LabelBinarizer()
        lb.fit(testing_labels)
        y_true = lb.transform(testing_labels)

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        pr_avg = {}
        roc_auc_temp = {}

        model_name_ = model_name
        if model_name == 'ANN':
            model_name_ = 0
        elif model_name == 'KNN':
            model_name_ = 1
        elif model_name == 'SVM':
            model_name_ = 2
        elif model_name == 'NB':
            model_name_ = 3
        else:
            model_name_ = 4

        pr_avg.setdefault(model_name_, [])
        roc_auc_temp.setdefault(model_name_, [])

        roc_auc_avg = {}
        roc_auc_avg.setdefault(model_name_, [])

        for i in range(len(lb.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predictions[:, 0])
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc_temp[model_name_].append(roc_auc[i])

        roc_auc_avg[model_name_].append(
            sum(roc_auc_temp[model_name_]) / len(roc_auc_temp[model_name_]))

        sum_ = 0
        for clss in list(fpr.keys()):
            sum_ += sum(fpr[clss])
        pr_avg[model_name_].append(sum_ / len(fpr))

        sum_ = 0
        for clss in list(tpr.keys()):
            sum_ += sum(tpr[clss])
        pr_avg[model_name_].append(sum_ / len(tpr))

        # print("pr", pr_avg)
        # print(roc_auc_avg)

        #  plot the roc curve for the model and save it to a file in the output directory
        plt.figure()
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve (class {lb.classes_[i]}, {model_name}) (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'ROC_{model_name}.png'))
        plt.close()

        fpr, tpr = {}, {}
        k, val = pr_avg.keys(), pr_avg.values()
        n_k, n_val = list(k), list(chain.from_iterable(list(val)))

        i, fp, tp = n_k[0], n_val[0], n_val[1]
        fpr.setdefault(i, [])
        tpr.setdefault(i, [])
        fpr[i].append(fp)
        tpr[i].append(tp)

        fpr, tpr, roc_auc = fpr, tpr, roc_auc_avg

        return fpr, tpr, roc_auc
