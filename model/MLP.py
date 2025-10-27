import torch
import torch.nn as nn

class MLP5(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP5, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size)
        self.batchnorm4 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm4(self.layer4(x)))
        x = self.dropout(x)
        x = self.layer5(x)
        return x

class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP3, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
   
class MLP_2(nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.layer1 = nn.Linear(512, 256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.batchnorm1(self.layer1(x)))
        x = self.relu2(self.batchnorm2(self.layer2(x)))
        x = self.layer3(x)
        return x
    
class MLP_3(nn.Module):
    def __init__(self):
        super(MLP_3, self).__init__()
        self.layer1 = nn.Linear(1024, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.batchnorm1(self.layer1(x)))
        x = self.layer2(x)
        return x

class MLP_4(nn.Module):
    def __init__(self):
        super(MLP_4, self).__init__()
        self.layer1 = nn.Linear(512, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.batchnorm1(self.layer1(x)))
        x = self.layer2(x)
        return x
    
# from torchsummary import summary
# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = MLP_4()
# model.to(device)

# # for name, parameters in model.named_parameters():  
# #     print(name, ';', parameters.size())


# # summary(model, input_size=(1, 1024))
# inputs = torch.randn(1, 1, 256).to(device)
# # writer = SummaryWriter(log_dir="runs/model_logs")
# # writer.add_graph(model, inputs)
# # writer.close()

# flops, params = profile(model, (inputs,))
# print('flops: ', flops, 'params: ', params)
