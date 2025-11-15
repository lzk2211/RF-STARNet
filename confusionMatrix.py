import os
import json
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from datetime import datetime
# 确保 matplotlib 在 plot 函数内部导入
# import matplotlib
# import matplotlib.pyplot as plt

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1  # 保持 [predicted, true]

    def summary(self):
        # (此函数保持不变)
        # calculate accuracy
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
        """
        --- 这是修改后的 plot 函数 ---
        """
        try:
            import matplotlib
            # 使用无界面后端，避免服务器/无显示环境问题
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Matplotlib unavailable, skip confusion matrix plot: {e}")
            return
            
        matrix = self.matrix
        print("Confusion Matrix (Raw Counts):\n", matrix)

        # --- 新增：创建更合适的图窗大小 ---
        # 根据类别数量调整图窗大小，避免文本重叠
        fig_size = max(8, self.num_classes * 0.8) 
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # --- 新增：对矩阵进行列归一化 (按 True Label) ---
        # 这将显示“在所有真实标签为X的样本中，分别有多少百分比被预测为了Y”
        sum_of_cols = matrix.sum(axis=0, keepdims=True)
        # 使用 np.nan_to_num 避免除以0（如果某个类没有样本）
        normalized_matrix = np.nan_to_num(matrix / sum_of_cols)

        # --- 修改：使用归一化矩阵绘制热图 ---
        # vmin=0, vmax=1 确保颜色范围在 0% 到 100% 之间
        im = ax.imshow(normalized_matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)

        # 设置x轴坐标label (True Labels)
        ax.set_xticks(range(self.num_classes))
        ax.set_xticklabels(self.labels, rotation=45, ha="right") # 调整对齐
        # 设置y轴坐标label (Predicted Labels)
        ax.set_yticks(range(self.num_classes))
        ax.set_yticklabels(self.labels)
        
        # --- 修改：为 Colorbar 添加标签 ---
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Column-Normalized Frequency (Recall %)')

        ax.set_xlabel('True Labels')
        ax.set_ylabel('Predicted Labels')
        # ax.set_title(folder_path + ' Confusion matrix') # 标题通常可以省略

        # --- 修改：在图中标注“计数”和“百分比”信息 ---
        # thresh = normalized_matrix.max() / 2 # 旧阈值
        thresh = 0.5 # 固定阈值，0.5以上用白色，以下用黑色
        
        for x in range(self.num_classes): # x 对应 True Label (column)
            for y in range(self.num_classes): # y 对应 Predicted Label (row)
                
                count = int(matrix[y, x])
                
                if count == 0:
                    info = "0" # 如果计数为0，只显示0
                else:
                    # 获取归一化百分比
                    percent = normalized_matrix[y, x] * 100
                    # 格式化文本：原始计数 \n (百分比)
                    info = f"{count}\n({percent:.1f}%)"
                
                # 根据背景色选择文字颜色
                text_color = "white" if normalized_matrix[y, x] > thresh else "black"
                
                # --- 新增：突出对角线文本 ---
                fontweight = "bold" if x == y else "normal"
                
                ax.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color=text_color,
                         fontsize=8, # 缩小字号以适应单元格
                         fontweight=fontweight) # 应用字体粗细

        fig.tight_layout()
        # 保存为新文件名，以区分
        save_path = os.path.join(folder_path, 'Confusion_matrix_normalized.png')
        fig.savefig(save_path)
        print(f"Normalized confusion matrix saved to {save_path}")
        plt.close(fig) # 关闭图窗，释放内存