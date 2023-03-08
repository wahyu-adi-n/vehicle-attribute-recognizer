from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import numpy as np
import torch


class ClassificationMetrics:
    def __init__(self):
        self.result_metrics = dict()

    def __convert_to_numpy_array(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        return (y_true, y_pred)

    def __calculate_metrics(self, y_true: np.array, y_pred: np.array, averages='macro') -> dict:
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(
            y_true=y_true, y_pred=y_pred, average=averages)
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=averages)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=averages)
        return {
            'precision': precision,
            'accuracy': acc,
            'recall': recall,
            'f1_score': f1,
        }

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, averages='macro') -> dict:
        y_true, y_pred = self.__convert_to_numpy_array(y_true, y_pred)
        self.result_metrics = self.__calculate_metrics(
            y_true=y_true, y_pred=y_pred, averages=averages)
        return self.result_metrics
