import os
import json
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from datetime import datetime

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        print("the model accuracy is ", acc)
        return acc, table

    def plot(self, folder_path):
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Matplotlib unavailable, skip confusion matrix plot: {e}")
            return
            
        matrix = self.matrix
        print("Confusion Matrix (Raw Counts):\n", matrix)

        fig_size = max(8, self.num_classes * 0.8) 
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sum_of_cols = matrix.sum(axis=0, keepdims=True)
        normalized_matrix = np.nan_to_num(matrix / sum_of_cols)

        im = ax.imshow(normalized_matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)

        ax.set_xticks(range(self.num_classes))
        ax.set_xticklabels(self.labels, rotation=45, ha="right") # 调整对齐
        ax.set_yticks(range(self.num_classes))
        ax.set_yticklabels(self.labels)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Column-Normalized Frequency (Recall %)')

        ax.set_xlabel('True Labels')
        ax.set_ylabel('Predicted Labels')
        thresh = 0.5
        
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                
                count = int(matrix[y, x])
                
                if count == 0:
                    info = "0"
                else:
                    percent = normalized_matrix[y, x] * 100
                    info = f"{count}\n({percent:.1f}%)"
                
                text_color = "white" if normalized_matrix[y, x] > thresh else "black"
                
                fontweight = "bold" if x == y else "normal"
                
                ax.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color=text_color,
                         fontsize=8,
                         fontweight=fontweight)

        fig.tight_layout()
        save_path = os.path.join(folder_path, 'Confusion_matrix_normalized.png')
        fig.savefig(save_path)
        print(f"Normalized confusion matrix saved to {save_path}")
        plt.close(fig)
        