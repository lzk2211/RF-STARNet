import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import vgg16, VGG16_Weights
import numpy as np

class MIML_DCNN(nn.Module):
    def __init__(self, num_classes):
        super(MIML_DCNN, self).__init__()
        
        # 输入调整层：将单通道图像转为3通道
        self.input_layer = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        
        # 加载预训练的VGG16，不包含顶层
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 移除最后的max pooling层(相当于原代码中移除最后4层)
        self.base_model = nn.Sequential(*list(vgg.features.children())[:-4])
        
        # 冻结前14层
        for i, param in enumerate(self.base_model.parameters()):
            if i < 28:  # VGG16每层有2个参数(权重和偏置)
                param.requires_grad = False
        
        # 额外的卷积层和批归一化
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024, momentum=0.8)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2048, momentum=0.8)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(2048, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(200, momentum=0.8)
        )
        
        # MIML的全连接层
        self.dense1 = nn.Linear(200, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.thresholds = nn.Parameter(torch.ones(num_classes) * 0.5)
    
    def forward(self, x, return_features=False):
        # 将单通道转为3通道
        x = self.input_layer(x)
        
        # 基础模型
        x = self.base_model(x)
        
        # 额外的卷积块
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 获取特征图的形状
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        num_instances = h * w  # 对于512x512的输入，这个值会比较大
        
        # 重塑特征图
        x = x.permute(0, 2, 3, 1).contiguous()  # [batch, h, w, 200]
        x = x.view(batch_size, num_instances, 200)  # [batch, h*w, 200]

        if return_features:
            return x  # [batch, h*w, 200]
        
        # 全连接层
        x = self.relu(self.dense1(x))  # [batch, h*w, 128]
        x = self.relu(self.dense2(x))  # [batch, h*w, 128]
        x = self.dense3(x)  # [batch, h*w, 4]
        
        # 实例间的最大池化
        x = x.view(batch_size, num_instances, 4, 1)  # [batch, h*w, 4, 1]
        x = x.permute(0, 3, 2, 1)  # [batch, 1, 4, h*w]
        x = nn.functional.max_pool2d(x, kernel_size=(1, num_instances))  # [batch, 1, 4, 1]
        x = x.view(batch_size, 4)  # [batch, 4]
        
        return x

# 训练函数
def train_model(model, train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-6)
    
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model

# 主执行代码
if __name__ == "__main__":
    # 加载数据
    # X_train = np.load('E:/MIML/X_train.npy')
    # Y_label_train = np.load('E:/MIML/Y_train.npy')
    
    # # 将numpy数组转换为PyTorch张量
    # X_train = torch.from_numpy(X_train).float()
    # Y_label_train = torch.from_numpy(Y_label_train).float()
    
    # # 创建数据集和数据加载器
    # dataset = TensorDataset(X_train, Y_label_train)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化并训练模型
    
    # print(model)  # 打印模型结构
    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MIML_DCNN()
    model = model.to(device)

    # summary(model, input_size=(1, 512, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)

    
    # trained_model = train_model(model, train_loader, val_loader, epochs=50)
    
    # # 保存模型
    # torch.save(trained_model.state_dict(), 'E:/MIML/MIML_model_epoch.pth')
    
    print("已创建模型架构。取消数据加载和训练代码的注释以训练模型。")