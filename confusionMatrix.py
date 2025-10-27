import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
# 延迟导入 matplotlib，避免在导入阶段触发 NumPy ABI 问题
from prettytable import PrettyTable
from datetime import datetime

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
            self.matrix[p, t] += 1

    def summary(self):
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
        try:
            import matplotlib
            # 使用无界面后端，避免服务器/无显示环境问题
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Matplotlib unavailable, skip confusion matrix plot: {e}")
            return
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        # plt.title(folder_path + ' Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(folder_path + '/' + 'Confusion matrix.png')
        # plt.show()


# if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # data_transform = transforms.Compose([transforms.Resize(256),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # image_path = '/root/autodl-tmp/lzk/D00/data_set'
    # assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
    #                                         transform=data_transform)

    # batch_size = 32
    # class_num = 10
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=False,
    #                                               num_workers=2)
    # net = ResNet10(num_classes=class_num)
    # # load pretrain weights
    # model_weight_path = "/root/autodl-tmp/lzk/test/ANN_model_deep.pth"
    # assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # net.to(device)

    # # read class_indict
    # json_label_path = './class_indices.json'
    # assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    # json_file = open(json_label_path, 'r')
    # class_indict = json.load(json_file)

    # labels = [label for _, label in class_indict.items()]
    # confusion = ConfusionMatrix(num_classes=class_num, labels=labels)
    # net.eval()
    # with torch.no_grad():
    #     for val_data in tqdm(validate_loader):
    #         val_images, val_labels = val_data
    #         outputs = net(val_images.to(device))
    #         outputs = torch.softmax(outputs, dim=1)
    #         outputs = torch.argmax(outputs, dim=1)
    #         confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    # confusion.plot()
    # confusion.summary()

